
# Table of Contents

1.  [Documentation](#orgbbae14f)
    1.  [Dynamics](#org7bec790)
        1.  [Currents](#orgde50508)
        2.  [Rates](#org1c33401)
    2.  [Connectivity](#orgad86fb8)



<a id="orgbbae14f"></a>

# Documentation

Neurotorch is an implementation of a rate based recurrent neural network trainer and simulator.


<a id="org7bec790"></a>

## Dynamics


<a id="orgde50508"></a>

### Currents

Neuron $i$ in population $A$ has a reccurent input $h^A_i$,

$$  \tau_{syn} \frac{dh_i}{dt}(t) = - h_i(t) + \sum_j J_{ij} h_j(t) $$

or not

$$ h^A_i(t) = \sum_{jB} J^{AB}_{ij} h_j(t) $$


<a id="org1c33401"></a>

### Rates

The models can have rate dynamics (setting **RATE<sub>DYN</sub>** to 1 in the configuration file):

$$ \tau_A \frac{d r^A_i}{dt}(t) = - r^A_i(t) + \Phi( \sum_{jB} J^{AB}_{ij} h^{AB}_j(t) + h^A_{ext}(t)) $$

Here, $r_i$ is the rate of unit $i$ in population $A$

otherwise rates will be instantaneous:

$$ r^A_i(t) = \Phi(\sum_{jB} J^{AB}_{ij} h_j(t) + h^A_{ext}(t)) $$

Here $\Phi$ is the transfer function defined in **src/activation.py** and can be set to a threshold linear, a sigmoid or a non linear function (Brunel et al., 2003).


<a id="orgad86fb8"></a>

## Connectivity

Probability of connection from population B to A:

1.  Sparse Nets

    by default it is a sparse net
    
    $$ P_{ij}^{AB} = \frac{K_B}{N_B} $$
    
    otherwise
    it can be cosine
    
    $$ P_{ij}^{AB} = ( 1.0 + \Kappa_B \cos(\theta_i^A - \theta_j^B) ) $$
    
    and also low rank
    
    $$ J_{ij}^{AB} = \frac{J_{AB}}{\sqrt{K_B}} with proba. P_{ij}^{AB} * \frac{K_B}{N_B} $$
    $$ 0 otherwise $$

2.  All to all

    $$ J_{ij}^{AB} =  \frac{J_{AB}}{N_B} P_{ij}^{AB} $$
    
    where Pij can be as above.

