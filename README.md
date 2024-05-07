# AnodeConsumption
Use level set method coupled with elettromagnetic equations to simulate the consumption of the anode

\[
-\nabla \cdot \sigma \nabla V = f \quad \text{in } \Omega
\]

\[
V = 1 \quad \text{in } \partial \Omega_{\text{in}}
\]

\[
V = -1 \quad \text{in } \partial \Omega_{\text{out}}
\]

\[
\sigma \nabla V \cdot \vec{n} = 0 \quad \text{in } \partial \Omega / (\partial \Omega_{\text{in}} \cup \partial \Omega_{\text{out}})
\]

\[
\{V\}_{\Gamma} = 0 \quad \text{on } \Gamma(t)
\]

\[
\{\sigma \nabla V \cdot \vec{n}\}_{\Gamma} = \vec{g} \quad \text{on } \Gamma(t)
\]


