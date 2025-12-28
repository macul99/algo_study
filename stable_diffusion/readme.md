# Study Stable Diffusion

### Background Formular
<img src="ddpm_background_formular.png" alt="background_formular" width="50%">
<img src="equation_5_to_7.png" alt="background_formular" width="50%">

### Proof of Equation (3)
<img src="formular_3_proof_1.png" alt="proof" width="50%">
<img src="formular_3_proof_2.png" alt="proof" width="50%">
<img src="formular_3_proof_3.png" alt="proof" width="50%">
<img src="formular_3_proof_4.png" alt="proof" width="50%">
<img src="formular_3_proof_5.png" alt="proof" width="50%">

### Proof of Equation (5)
- Note, by condition on X0, equation 5 is tractable!

<img src="proof_equation_5.png" alt="proof" width="50%">

### Proof of Equation (7)
<img src="proof_equation_7_1.png" alt="proof" width="50%">
<img src="proof_equation_7_2.png" alt="proof" width="50%">

-Note: we need to find \miu and \sigma of x_t-1 which is (x-miu)^2/sigma^2 = x^2/sigma^2-2*miu*x/sigma
as shown in below step, we find sigma first, then followed by miu

<img src="proof_equation_7_3.png" alt="proof" width="50%">
<img src="proof_equation_7_4.png" alt="proof" width="50%">


### Reparameterization Trick
<img src="reparameterization_track_1.png" alt="proof" width="50%">
<img src="reparameterization_track_2.png" alt="proof" width="50%">
reparameterization trick fomular will be used to substitute x0 in equation (7) of the paper to get u(xt, x0) as in equation (10)


### transition from Equation 7 to the actual training objective equation (12)
<img src="training_objective_1.png" alt="proof" width="50%">
<img src="training_objective_2.png" alt="proof" width="50%">


### The Sampling Loop (Algorithm 2)
<img src="algo_1_and_2.png" alt="proof" width="50%">
<img src="algo_2_1.png" alt="proof" width="50%">
<img src="algo_2_2.png" alt="proof" width="50%">
<img src="algo_2_3.png" alt="proof" width="50%">

