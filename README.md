# AnodeConsumption
Use level set method coupled with elettromagnetic equations to simulate the consumption of the anode

<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

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


