import numpy as np
from scipy import sparse
import pyfem
import utils

### Construct a sample problem
# Set parameters of problem
nelemsx = 5
nelemsy = 5
ndims = 2
nelems = nelemsx * nelemsy
neigs = 12
nnodes = (nelemsx + 1) * (nelemsy + 1)
nnodes_per_elem = 2 ** ndims
nquads = nnodes_per_elem

# Format translator functions
def global_to_nodal(data_g):
    "Reshape global-DOF data into nodal-DOF data. (data_g -> data_n)"
    return data_g.reshape((nnodes, ndims) + data_g.shape[1:])


def nodal_to_global(data_n):
    "Reshape nodal-DOF data into global-DOF data. (data_g -> data_n)"
    return data_n.reshape((ndims * nnodes,) + data_n.shape[2:])


def nodal_to_elemental(data_n, conn):
    "Scatter nodal-DOF data to element nodes. (data_n -> data_e)"
    return data_n[conn, ...]


def elemental_to_nodal(data_e, conn):
    "Aggregate element-nodal data to nodal-DOF data. (data_e -> data_n)"
    data_n = np.zeros((nnodes, ndims) + data_e.shape[3:], data_e.dtype)
    np.add.at(data_n, conn, data_e)
    return data_n


def flatten_elemental(data_e):
    "Crunch element-wise nodal data into element-wise DOF data. (data_e -> data_f)"
    return data_e.reshape((nelems, ndims * nnodes_per_elem) + data_e.shape[3:])


def expand_elemental(data_f):
    "Expand element-wise DOF data into element-wise nodal data. (data_f -> data_e)"
    return data_f.reshape((nelems, nnodes_per_elem, ndims) + data_f.shape[2:])


# Generate quadrature and weighting scheme
quadrature = pyfem.QuadratureBilinear2D()
wq = quadrature.get_weight()

# Generate basis and logical shape derivatives
basis = pyfem.BasisBilinear2D(quadrature)
Nderiv = basis.eval_shape_fun_deriv()

# Build connectivity matrix
conn = np.zeros((nelems, nnodes_per_elem), int)
for x in range(nelemsx):
    for y in range(nelemsy):
        iE = y + x * nelemsy
        iN = y + x * (nelemsy + 1)
        conn[iE, :] = [iN, iN + 1, iN + nelemsy + 1, iN + nelemsy + 2]

# Construct element physical locations
X_g = np.zeros((ndims * nnodes))
for x in range(nelemsx + 1):
    for y in range(nelemsy + 1):
        iN = y + x * (nelemsy + 1)
        X_g[2 * iN] = x
        X_g[2 * iN + 1] = y

# Convert element physical locations to other necessary forms
X_n = global_to_nodal(X_g)
X_e = nodal_to_elemental(X_n, conn)

# Construct simple boundary and loading conditions
dof_fixed = np.arange(ndims * (nelemsx + 1))
nodal_force = {nnodes - 1: [0, 1]}

# Declare material properties and penalization factor
E = 10
nu = 0.3
p = 3

# Initialize PyFEM model
model = pyfem.LinearElasticity(
    X_n, conn, dof_fixed, None, nodal_force, quadrature, basis, E, nu, p
)

# Triangular number generator
def tri(n):
    "Compute the n-th triangular number, as an integer: ex. `tri(3) == 6`"
    return int(n * (n + 1) / 2)


# Material property penalization
def material_penalty(xe, p):
    """
    Compute RAMP penalization of ELEMENT-wise design variables xe.
    Compare to: `pyfem.LinearElasticity._update_material_property(self, rho)`.

    Inputs:
        xe: Element design variable vector, (nelems,)
        p: Penalization factor, scalar

    Output:
        se: Element penalized stiffnesses, (nelems,)
    """
    return xe / (1 + p * (1 - xe))


def material_penalty_deriv(xe, p):
    """
    Compute RAMP penalization derivative of ELEMENT-wise design variables xe.
    Compare to: `pyfem.LinearElasticity._update_material_property_deriv(self, rho)`.

    Inputs:
        xe: Element design variable vector, (nelems,)
        p: Penalization factor, scalar

    Output:
        dse: Element penalized stiffness derivative vector, (nelems,)
    """
    return (1 + p) / (1 + p * (1 - xe)) ** 2


# Matrix problem solvers
def solve(K_gg, f_g):
    """
    Solve the displacement of the system under load f and subject to boundary conditions.
        Compare to: `pyfem.LinearElasticity.compliance(xe)[0]`

    Inputs:
        K_gg: Global stiffness matrix, (ndims*nnodes, ndims*nnodes)
        f_g: Global force vector, (ndims*nnodes,)

    Output:
        u_g: Global displacement field, (ndims*nnodes,)
    """
    KD_gg, fD_g = model.apply_dirichlet_bcs(K_gg, f_g, enforce_symmetric_K=True)
    u_g = sparse.linalg.spsolve(KD_gg, fD_g)
    return u_g


def eigsolve(K_gg, G_gg):
    """
    Solve the buckling eigenvalue problem (G+μK)φ=0.

    Inputs:
        K_gg: Global stiffness matrix, (ndims*nnodes, ndims*nnodes)
        G_gg: Global tangent stiffness matrix, (ndims*nnodes, ndims*nnodes)

    Outputs:
        mu: Eigenvalues, (neigs,)
        phi_g: Global eigenmode displacement vectors, (ndims*nnodes, neigs)
    """
    rhs_g = np.zeros(ndims * nnodes)
    KD_gg = model.apply_dirichlet_bcs(K_gg, rhs_g, enforce_symmetric_K=True)[0]
    GD_gg = model.apply_dirichlet_bcs(G_gg, rhs_g, enforce_symmetric_K=True)[0]

    # Compute and process eigensolutions
    eivals, eivecs_g = sparse.linalg.eigsh(GD_gg, k=neigs + 4, M=KD_gg, which="SA")
    ii = np.argsort(eivals)
    mu = -eivals[ii[:neigs]]
    eivecs_g = eivecs_g[:, ii[:neigs]]
    phi_g = eivecs_g / np.sqrt((eivecs_g.T * KD_gg @ eivecs_g).diagonal())
    return mu, phi_g


# Core FSB algorithm implementation
def compute_elem_compact_tangent(model, u_g, wq):
    """
    Compute nonlinear element tangent stiffness matrix in compact form Ze.
    This requires expansion via `Ge = expand_elem_compact_tangent(Ze)`,
    and then assembly via `pyfem.LinearElasticity._assemble_jacobian(self, Ge)`.

    Inputs:
        from `pyfem.LinearElasticity`:
            Be: Element linear strain-displacement operator, (nelems, nquads, tri(ndims), ndims*nnodes_per_elem)
            C0: Linear elastic constitutive matrix, (tri(ndims), tri(ndims))
            detJq: Quadrature Jacobian determinant, (nelems, nquads)
            Ngrad: Physical shape function derivatives, (nelems, nquads, nnodes_per_elem, ndims)
        from `model`:
            beta: Dual deformation gradient, (nelems, nquads, tri(ndims), tri(nnodes_per_elem))
            gamma: Strain coefficient, (tri(ndims),)
            P1: 1st semistrain operator (nelems, nquads, ndims, tri(ndims), tri(nnodes_per_elem))
            P2: 2nd semistrain operator (nelems, nquads, ndims, tri(ndims), tri(nnodes_per_elem))
        u_g: Global displacement solution, (nnodes_per_elem*ndims,)
        wq: Quadrature weights, (nquads,)
            Compare to: `pyfem.LinearElasticity.wq`

    Output:
        Ze: Compact element tangent stiffness matrix, (nelems, tri(nnodes_per_elem))
    """

    # Scatter displacement to elements
    u_f = flatten_elemental(nodal_to_elemental(global_to_nodal(u_g), conn))

    # Compute strain at all elements and quadrature points
    Ee = np.einsum("eqtd, ed -> eqt", model.Be, u_f)
    Ee += np.einsum(
        "t, eqxtc, ec, eqxtd, ed -> eqt",
        model.gamma,
        model.P1,
        u_f,
        model.P2,
        u_f,
        optimize=True,
    )

    # Compute stress at all elements and quadrature points
    Se = np.einsum("st, eqt -> eqs", model.C0, Ee)

    # Integrate compact tangent stiffness matrix
    Ze = np.einsum(
        "eq, q, eqs, eqsz -> ez", model.detJq, wq, Se, model.beta, optimize=True
    )

    return Ze


def expand_elem_compact_tangent(model, Ze, xe):

    """
    Expand the compact element tangent stiffness representation Ze returned by
    `compute_elem_compact_tangent(...)` into the full elemental tangent stiffness matrix Ge,
    which can then be assembled with pyfem.LinearElasticity._assemble_jacobian(self, Ge)`.

    Inputs:
        Ze: Compact element tangent stiffness matrix, (nelems, tri(nnodes_per_elem))
        xe: ELEMENT-wise design variables, (nelems,)

    Outputs:
        Ge: Expanded element tangent stiffness matrix, (nelems, nnodes_per_elem, nnodes_per_elem)
    """

    # Retrieve material penalization factor for each element
    sG = material_penalty(xe, model.p + 1)

    # Unfold Ze into ge
    ge = np.zeros((nelems, nnodes_per_elem, nnodes_per_elem))
    ge[:, model.l[:, 0], model.l[:, 1]] = np.einsum("e, ez -> ez", sG, Ze)
    ge += np.triu(ge.transpose((0, 2, 1)), k=1)

    # Expand ge into Ge
    Ge = np.kron(ge, np.eye(ndims))

    return Ge


def compute_buck_eigen_sensitivity(model, u_g, wq, xe, Ze, K_gg, Ke_ff, mu, phi_g):
    """
    Compute sensitivities of buckling load factors to design variable changes.

    Inputs:
        from `pyfem.LinearElasticity`:
            Be: Element linear strain-displacement operator, (nelems, nquads, tri(ndims), ndims*nnodes_per_elem)
            conn: Connectivity matrix, (nelems, nnodes_per_elem)
            C0: Linear elastic constitutive matrix, (tri(ndims), tri(ndims))
            detJq: Quadrature Jacobian determinant, (nelems, nquads)
            Ngrad: Physical shape function derivatives, (nelems, nquads, nnodes_per_elem, ndims)
            p: Material penalization factor, scalar
        from `model`:
            beta: Dual deformation gradient, (nelems, nquads, tri(ndims), tri(nnodes_per_elem))
            gamma: Strain coefficient, (tri(ndims),)
            P1: 1st semistrain operator (nelems, nquads, ndims, tri(ndims), tri(nnodes_per_elem))
            P2: 2nd semistrain operator (nelems, nquads, ndims, tri(ndims), tri(nnodes_per_elem))
        u_g: Global displacement solution, (nnodes_per_elem*ndims,)
        wq: Quadrature weights, (nquads,)
            Compare to: `pyfem.LinearElasticity.wq`
        xe: ELEMENT-wise design variables, (nelems,)
        Ze: Compact element tangent stiffness representation returned by `compute_elem_compact_tangent(...)`
        K_gg: Global sparse stiffness matrix, (ndims*nnodes, ndims*nnodes)
            Compare to: `pyfem.LinearElasticity.compute_jacobian(rho)`
        Ke_ff: Element stiffness matrices, (nelems, ndims*nnodes_per_elem, ndims*nnodes_per_elem)
            Compare to: `pyfem.LinearElasticity.compute_element_jacobian(rho)`
        mu: Eigenvalues of the buckling problem, (neigs,)
        phi_g: Eigenvectors of the buckling problem, (ndims*nnodes, neigs)

    Output:
        dmudxe: Sensitivities of buckling eigenvalues to design variables, (nelems, neigs)
    """

    u_f = flatten_elemental(nodal_to_elemental(global_to_nodal(u_g), model.conn))

    # Compute element strain sensitivity to displacement solution
    dEdu = (
        model.Be
        + np.einsum(
            "t, eqxtd, eqxtc, ec -> eqtd",
            model.gamma,
            model.P2,
            model.P1,
            u_f,
            optimize=True,
        )
        + np.einsum(
            "t, eqxtd, eqxtc, ec -> eqtd",
            model.gamma,
            model.P1,
            model.P2,
            u_f,
            optimize=True,
        )
    )

    # Compute element stress sensitivity to displacement solution
    dSdu = np.einsum("st, eqtd -> eqsd", model.C0, dEdu)

    # Compute sensitivity of compact element tangent stiffness to displacement solution
    dZdu = np.einsum(
        "eq, q, eqsd, eqsz -> edz", model.detJq, wq, dSdu, model.beta, optimize=True
    )

    # Compute material penalties and derivatives
    sG = material_penalty(xe, model.p + 1)
    dsK = material_penalty_deriv(xe, model.p)
    dsG = material_penalty_deriv(xe, model.p + 1)

    # Compute dual eigenvector product terms
    phi_e = nodal_to_elemental(global_to_nodal(phi_g), conn)
    phi_f = flatten_elemental(phi_e)
    pe = np.einsum(
        "ezxg, ezxg -> ezg", phi_e[:, model.l[:, 0], ...], phi_e[:, model.l[:, 1], ...]
    )

    # Construct and solve adjoint problem
    adjLoad_f = np.einsum("e, ezg, edz -> edg", sG, pe, dZdu)
    adjLoad_g = nodal_to_global(
        elemental_to_nodal(expand_elemental(adjLoad_f), model.conn)
    )
    adjDisp_g = solve(K_gg, adjLoad_g)
    adjDisp_f = flatten_elemental(
        nodal_to_elemental(global_to_nodal(adjDisp_g), model.conn)
    )

    # Assemble eigenvalue sensitivity
    dmudxe = -np.einsum("e, z, ezg, ez -> eg", dsG, model.alpha, pe, Ze)
    dmudxe -= np.einsum("g, e, ecg, ecd, edg -> eg", mu, dsK, phi_f, Ke_ff, phi_f)
    dmudxe += np.einsum("e, ec, ecd, edg -> eg", dsK, u_f, Ke_ff, adjDisp_f)

    return dmudxe


# Compute element logical-physical transformation
model.Jq = np.zeros((nelems, nquads, ndims, ndims))
utils.compute_jtrans(X_e, Nderiv, model.Jq)

# Compute determinant of element Jacobian
model.detJq = np.zeros((nelems, nquads))
utils.compute_jdet(model.Jq, model.detJq)

# Compute inverse transform and physical shape function derivatives
model.invJq = np.zeros((nelems, nquads, ndims, ndims))
model.Ngrad = np.zeros((nelems, nquads, nnodes_per_elem, ndims))
utils.compute_basis_grad(model.Jq, model.detJq, Nderiv, model.invJq, model.Ngrad)

# Compute linear strain-displacement operator
model.Be = np.zeros((nelems, nquads, tri(ndims), ndims * nnodes_per_elem))
model._compute_element_Bmat(model.Ngrad, model.Be)

# Assign random design vector
xe = np.random.rand(nelems)

# Compute penalized stiffness matrix
# Cq must be broadcast to all quadrature points, as it is constant inside the element
Cq = np.tile(material_penalty(xe, p), (nquads, 1)).transpose()
Ke_ff = np.zeros((nelems, ndims * nnodes_per_elem, ndims * nnodes_per_elem))
model._einsum_element_jacobian(model.detJq, wq, model.Be, Cq, model.C0, Ke_ff)
K_gg = model._assemble_jacobian(Ke_ff)

# Solve element operating condition
u_g = solve(K_gg, model.compute_rhs())

# Compute FSG condensed parameters
model.l = np.zeros((tri(nnodes_per_elem), 2), int)
for i in range(nnodes_per_elem):
    lrange = range(-tri(i + 1), -tri(i))
    model.l[lrange, 0] = range(nnodes_per_elem - i - 1, nnodes_per_elem)
    model.l[lrange, 1] = nnodes_per_elem - 1 - i
model.alpha = np.ones(tri(nnodes_per_elem)) + (model.l[:, 0] != model.l[:, 1])
model.beta = np.stack(
    (
        model.Ngrad[..., model.l[:, 0], 0] * model.Ngrad[..., model.l[:, 1], 0],
        model.Ngrad[..., model.l[:, 0], 1] * model.Ngrad[..., model.l[:, 1], 1],
        model.Ngrad[..., model.l[:, 0], 0] * model.Ngrad[..., model.l[:, 1], 1]
        + model.Ngrad[..., model.l[:, 0], 1] * model.Ngrad[..., model.l[:, 1], 0],
    ),
    -2,
)

# Compute nonlinear FSG strain parameters
model.gamma = [0.5, 0.5, 1]

PI = np.kron(
    np.expand_dims(model.Ngrad.transpose(0, 1, 3, 2), axis=3),
    np.expand_dims(np.eye(ndims), axis=(0, 1, 2)),
)
model.P1 = np.concatenate(
    (
        PI,
        np.kron(
            np.expand_dims(model.Ngrad[..., 0], axis=(2, 3)),
            np.expand_dims(np.eye(ndims), axis=(0, 1, 3)),
        ),
    ),
    axis=3,
)
model.P2 = np.concatenate(
    (
        PI,
        np.kron(
            np.expand_dims(model.Ngrad[..., 1], axis=(2, 3)),
            np.expand_dims(np.eye(ndims), axis=(0, 1, 3)),
        ),
    ),
    axis=3,
)

# Compute tangent stiffness matrix
Ze = compute_elem_compact_tangent(model, u_g, wq)
Ge_ff = expand_elem_compact_tangent(model, Ze, xe)
G_gg = model._assemble_jacobian(Ge_ff)

# Compute buckling eigenvalues and load factors
mu, phi_g = eigsolve(K_gg, G_gg)
BLF = 1 / mu

# Compute buckling eigenvalue and load factor sensitivities
dmudxe = compute_buck_eigen_sensitivity(model, u_g, wq, xe, Ze, K_gg, Ke_ff, mu, phi_g)
dBLFdxe = -dmudxe * BLF ** 2
