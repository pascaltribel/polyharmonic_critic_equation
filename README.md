# polyharmonic_critic_equation

## Main objective : 
Find a PINN-inspired way to obtain **nontrivial** solutions to *elliptic* *nonlinear* equations in arbitrary domains and dimensions. We aim at obtaining solutions $u$ to problems of the form

$$L u = f(u),$$

$L$ is an elliptic operator, for instance $L = -\Delta$ the Laplacian, $f$ is a real-valued function, which is nonlinear, for instance $f(u) = \lambda u + |u|^{p-2}u$, for $\lambda \in \mathbb{R}$ and $p>2$. The typical equation we wish to study is 

$$-\Delta u = \lambda u + |u|^{p-2}u.$$

## Motivation from nonlinear analysis :
When $n\geq 3$, $p= 2^* := \frac{2n}{n-2}$, the problem is said to be *critical*, the symmetries of the equation allow for sequences of solutions that converge weakly to 0, but which concentrate at some points. This is one of the easiest toy-model of what happens in cases where standard analysis methods stop working. 

It is known that all the positive solutions of the problem 

$$-\Delta u = |u|^{2^*-2}u \quad \text{on }\mathbb{R}^n \tag{1}$$

have a specific shape, there is $\mu>0$ and $z\in \mathbb{R}^n$ such that $u(x) = \mu^{\frac{n-2}{2}} \Big(\mu^2 + \frac{|x-z|^2}{\sqrt{n(n-2)}}\Big)^{\frac{n-2}{2}}$. Notice that all these solutions are obtained by transformation of the so-called standard bubble, $u(x) = \mu^{-\frac{n-2}{2}} B\big(\frac{x-z}{\mu}\big)$, where 

$$B(x) := \Big(1+ \frac{|x|^2}{\sqrt{n(n-2)}}\Big)^{\frac{n-2}{2}}$$

On the other hand, *sign-changing* solutions are far less understood, and while it is known that they have at least twice the energy of a positive one, i.e. that there exists $E_0>0$ such that 

$$\|u\|_{L^{2^*}} \geq 2 \|B\|_{L^{2^*}} + E_0 \quad \text{for all sign-changing solution $u$},$$

it is not known what the minimal ones look like. Moreover, besides some symmetric solutions obtained by constructive methods, or abstract existence results, we do not have many examples of what their profiles are like. It is possible to show that using the stereographic projection of the sphere $\mathbb{S}^n$ onto $\mathbb{R}^n$, equation [(1)](#motivation-from-nonlinear-analysis-) is equivalent to solving 

$$-\Delta_{Sph}u + \frac{n(n-2)}{4} u = |u|^{2^*-2}u \quad \text{on }\mathbb{S}^n.$$
 
The latter has the advantage of being defined on a compact surface $\mathbb{S}^n$. 

Similarly, there have been many investigations in the existence, uniqueness, multiplicity of solutions properties for equations of the form 
$$\left\{ \begin{aligned} 
    -\Delta u &= \lambda u + |u|^{2^*-2}u & & \text{on }\Omega \\
    u &= 0 & &\text{on } \partial \Omega,\end{aligned}\right.$$

where $\Omega$ is a bounded open set of $R^n$. When $\Omega$ is for instance the ball $B_n := \{x\in \mathbb{R}^n \,:\, |x| < 1\}$, we know that there exists a positive solution if and only if $\lambda \in (\lambda_*,\lambda_1)$, where $\lambda_1$ the first eigenvalue of the Laplacian in $B_n$, and $\lambda_* = \begin{cases}\frac{\lambda_1}{4} & \text{if } n = 3 \\ 0 & \text{if }n\geq 4\end{cases}$, and there are no nontrivial solutions when $\lambda\leq 0$. Obtaining explicit *sign-changing* solutions for arbitrary $\lambda >0$ is a hard problem.

Both of these examples show the significance of obtaining a method to find nontrivial solutions to critical elliptic equations, and more generally to nonlinear equation. When $n=2$ and $p>2$, or $n\geq 3$ and $2<p<2^*$, the problem is said to be *sub-critical*: Sequences of solutions remain compact, and existence result can be obtained by classical variational methods that have been developed since the 60's in the field of nonlinear analysis. It is however rare that explicit profiles of solutions can be described.

## The PINN approach :
We train a neural network $nn$ to minimise a given *loss* determined by the problem at hand, in this case 

$$\mathcal{L}(nn) = \int \Big[\lambda\, nn(x) + |nn(x)|^{p-2}nn(x) + \Delta nn(x) \Big]^2 dx$$

the "integral" is computed by taking a mean among a given discrete set of points. We wish to only minimise this loss function, while all the other conditions (periodicity in the sphere case, boundary conditions, ...) are strongly enforced by the model. The hard part consists in handling the properties of nonlinear elliptic PDEs while staying away from the trivial solution $u \equiv 0$.  

## Table of content of the files :
- [`equation_polyharmonique_critique`](/equation_polyharmonique_critique.ipynb) is the first attempt at the problem, set in a non-compact domain. This led us to choose to focus first on compact domains, as the choice of the behaviour of the function, and its inforcement, at infinity is not so clear.
- In [`polynom_fx`](/polynom_fx.ipynb) we were looking for specific functions $f$ so that $u = f\cdot nn$ satisfies some boundary conditions as well as behavior at infinity.

We started by considering the linear problem $u'' = \alpha\, u$ on the circle, which is the most basic domain where periodicity can represent a challenge.
- In [`Boule`](/Boule.ipynb) we explore how a trained PINN differs from true solutions (that are explicit and can be computed in this basic example), and specifically the distribution of the loss for various trainings.
- [`closeness`](/closeness.ipynb) compares the loss of a PINN which is trained to solve an ODE versus a NN that is trained to look pointwise like a known solution of the ODE.
- In [`Equation_cercle`](Equation_cercle.ipynb) we develop the basic ODE theory going from the linear problem $u'' = \alpha \,u$ to the nonlinear problem $u'' = u|u|^{p-2}$. In this ODE case, while we loose the explicit solutions, we are still able to describe entirely the phase space of the problem.
- [`Heart`](/Heart.ipynb) is an adaptation of `Boule` to the nonlinear ODE problem. 
- [`FD_solver`](/FD_solver.ipynb) is a finite-difference solver implementation of the previous two ODE problems.  
- In [`biharmonic_circle`](/biharmonic_circle.ipynb), we try to approximate a diffeomorphism of a topologically correct initial candidate to deform it towards a correct solution by minimizing the residuals of the equation.  

In a second step, we consider the nonlinear PDE $-\Delta_{Sph} u = u|u|^{p-2}$ on the sphere $\mathbb{S}^2$. 
- In [`polyharm_2D`](/polyharm_2D.ipynb) we look for nontrivial solutions by enforcing a condition $u(x_0) = u_0 >0$ at some $x_0 \in \mathbb{S}^2$. 

The question now becomes: How can one be sure that a given, trained, "solution" is indeed *close* to a true solution, in particular in cases as the sphere where periodicity conditions are hard to enforce.

## Current work
We now focus on the toy-model 

$$\begin{cases}
        -\Delta u = \lambda u + |u|^{p-2}u & \text{ on } B_2\\
        u = 0 & \text{ on } \partial B_2,
    \end{cases}$$

we look for solutions of the form 

$$u(x) = P(x) \, u_0(F(x)),$$  

where $u_0$ is a given "basis function" that satisfies the boundary condition and has some specific "topological" properties, $P : B_2 \to \mathbb{R}^+_0$ is a positive function, $F : B_2 \to B_2$ is a diffeomorphism (bijective continuous and differentiable) such that $F(\partial B_2) = \partial B_2$. We then train $nn$ to find $P$ and $F$, so that $u$ solves the equation, and satisfies the boundary conditions as well as the same topological properties as $u_0$ by construction.  
