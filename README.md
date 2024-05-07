# AnodeConsumption
Use level set method coupled with elettromagnetic equations to simulate the consumption of the anode

$$
\begin{aligned}
-\nabla \cdot \sigma \nabla V &= f && \text{in } \Omega, \\
V &= 1 && \text{in } \partial \Omega_{\text{in}}, \\
V &= -1 && \text{in } \partial \Omega_{\text{out}}, \\
\sigma \nabla V \cdot \vec{n} &= 0 && \text{in } \partial \Omega / (\partial \Omega_{\text{in}} \cup \partial \Omega_{\text{out}}), \\
\{V\}_{\Gamma} &= 0 && \text{on } \Gamma(t), \\
\{\sigma \nabla V \cdot \vec{n}\}_{\Gamma} &= \vec{g} && \text{on } \Gamma(t).
\end{aligned}
$$

