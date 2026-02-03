# MHD Xu-Auditor — Technical Notes

These notes summarize the diagnostic quantities produced by **MHD Xu-Auditor** for 2D incompressible resistive MHD in vorticity–potential form.

## Governing quantities

Let the stream function be $\phi(x,y,t)$ and the magnetic potential be $A(x,y,t)$. The velocity and magnetic fields are:
$$\mathbf{u}=\nabla^\perp\phi=(\partial_y\phi, -\partial_x\phi), \qquad \mathbf{B}=\nabla^\perp A=(\partial_y A, -\partial_x A)$$

Vorticity and current density are:
$$\omega = \zeta = -\Delta \phi, \qquad j = -\Delta A$$

## Modal energy and shell spectrum

In Fourier space, the kinetic + magnetic modal energy is assembled from potentials as:
$$E_{\mathbf{k}} = \frac{1}{2}|\mathbf{k}|^2\big(|\hat{\phi}_{\mathbf{k}}|^2 + |\hat{A}_{\mathbf{k}}|^2\big), \qquad \hat{\phi}_{\mathbf{k}} = -\frac{\hat{\omega}_{\mathbf{k}}}{|\mathbf{k}|^2} \ (\mathbf{k}\neq 0)$$

The (isotropic) shell spectrum $E(k)$ is formed by binning modes with $|\mathbf{k}|$ into integer shells.

## High-frequency diagnostics

### Tail energy
A robust tail metric integrates energy beyond a cutoff $k_c$:
$$E_{\text{tail}}(k_c) = \sum_{k \ge k_c} E(k)$$

### Blocking Index (spectral blocking)
Spectral blocking is quantified by comparing energy near the cutoff to a mid-band below it.
- **Edge band**: $[0.9 k_c, 1.0 k_c]$
- **Mid band**: $[0.55 k_c, 0.65 k_c]$

$$\mathrm{BI}(k_c)=\frac{\langle E(k)\rangle_{\text{edge}}}{\langle E(k)\rangle_{\text{mid}}}$$

As a rule of thumb for the Orszag–Tang stress tests in this project, values $\mathrm{BI} \gtrsim 2.4$ indicate pronounced high-$k$ pile-up.

### Regularity ratio ($\rho$)
The solver-side diagnostics use a regularity ratio $\rho(t)$ based on quadratic invariants built from vorticity. In the public auditor package, $\rho$ is computed with safeguards (avoiding log of zeros and tiny denominators) and is exported to `metrics.csv`.
