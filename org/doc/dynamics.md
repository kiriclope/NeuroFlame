
# Table of Contents

1.  [Documentation](#orgdb2087d)
    1.  [Dynamics](#org39faaf1)
        1.  [Currents](#orgd3c8ed3)
        2.  [Rates](#org57aa480)
    2.  [Connectivity](#orgf667d98)



<a id="orgdb2087d"></a>

# Documentation

Neurotorch is an implementation of a rate based recurrent neural network trainer and simulator.


<a id="org39faaf1"></a>

## Dynamics


<a id="orgd3c8ed3"></a>

### Currents

Neuron $i$ in population $A$ has a reccurent input $h^A_i$,

$$  \tau_{syn} \frac{dh_i}{dt}(t) = - h_i(t) + \sum_j J_{ij} h_j(t) $$

or not

$$ h^A_i(t) = \sum_{jB} J^{AB}_{ij} h_j(t) $$


<a id="org57aa480"></a>

### Rates

The models can have rate dynamics (setting **RATE<sub>DYN</sub>** to 1 in the configuration file):

$$ \tau_A \frac{d r^A_i}{dt}(t) = - r^A_i(t) + \Phi( \sum_{jB} J^{AB}_{ij} h^{AB}_j(t) + h^A_{ext}(t)) $$

\begin{equation}
\tau_A \frac{d r^A_i}{dt}(t) = - r^A_i(t) + \Phi( \sum_{jB} J^{AB}_{ij} h^{AB}_j(t) + h^A_{ext}(t))
\end{equation}

Here, $r_i$ is the rate of unit $i$ in population $A$

otherwise rates will be instantaneous:

\begin{equation}
  r^A_i(t) = \Phi(\sum_{jB} J^{AB}_{ij} h_j(t) + h^A_{ext}(t))
\end{equation}

Here $\Phi$ is the transfer function defined in **src/activation.py**


<a id="orgf667d98"></a>

## Connectivity

Probability of connection from population B to A:

1.  Sparse Nets

    by default it is a sparse net
    
    \begin{equation}
    P_{ij}^{AB} = \frac{K_B}{N_B}
    \end{equation}
    
    otherwise
    it can be cosine
    
    \begin{equation}
    P_{ij}^{AB} = ( 1.0 + \KAPPA_B \cos(\theta_i^A - \theta_j^B) )
    \end{equation}
    
    and also low rank
    
    \begin{equation}
      J_{ij}^{AB} = \frac{J_{AB}}{\sqrt{K_B}} with proba. P_{ij}^{AB} * \frac{K_B}{N_B} 
                   0 otherwise
    \end{equation}

2.  All to all

    \begin{equation}
      J_{ij}^{AB} =  \frac{J_{AB}}{N_B} P_{ij}^{AB}
    \end{equation}
    
    where Pij can be as above.

