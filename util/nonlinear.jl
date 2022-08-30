# Here is an example of creating custom node for nonlinear function
using ReactiveMP, Distributions, Random, BenchmarkTools, Rocket, GraphPPL, StableRNGs

# I will use ForneyLab as a backup implementation, no need to reinvent the wheel
# We are in the process of porting all of these rules into ReactiveMP
import ForneyLab

abstract type AbstractApproximationMethod end

# Unscented transform performs poorly on arbitray non-linear function
# TODO: alpha, beta and kappa cannot be passed to ForneyLab rule directly.
struct Unscented <: AbstractApproximationMethod
    alpha :: Float64
    beta  :: Float64
    kappa :: Float64
end

# Default constructor with default parameters
Unscented() = Unscented(
    ForneyLab.default_alpha, 
    ForneyLab.default_beta, 
    ForneyLab.default_kappa
) 

# Node creation 
struct DeltaNode end # Dummy structure just to make Julia happy

# Meta object that controls approximation methods and nonlinear node context
# We use meta objects to change what message update rule should be executed in ReactiveMP 
# By default most of the nodes in reactive do not require any meta-information
# But for the nonlinear node at least one essential piece of meta information is the function itself 
# In this example I use meta object to pass to the update rule 3 pieces of information:
#  1. nonlinear function, 2. its inverse (if available) and 3. approximation method 
struct DeltaNodeMeta{F, I, A}
    fn            :: F # Nonlinear function, for simplicity, we assume 1 input - 1 ouput
    fn_inv        :: I # Inverse of nonlinear function, put `nothing` if not available
    approximation :: A # Approximation method 
end

# Here we use a `@node` macro to create an 1input-1output nonlinear node
@node DeltaNode Deterministic [ out, in ]

# This line says that we want to request an inbound message 
# when computing backward (towards 'in', index '2') rule 
# (nothing,) means that we do not want to pre-initialise inbound message
# Instead we could pre-initialise inbound message with `(NormalMeanPrecision(0.0, 0.01), )` for example
ReactiveMP.default_functional_dependencies_pipeline(::Type{<:DeltaNode}) =
    ReactiveMP.RequireInboundFunctionalDependencies((2,), (NormalMeanPrecision(0.0, 0.01),))

# We need to define two Sum-product message computation rules for our new custom node
# - Rule for outbound message on `out` edge given inbound message on `in` edge
# - Rule for outbound message on `in` edge given inbound message on `in` and `out` edges
# - Both rules accept optional meta object

# Rule for outbound message on `out` edge given inbound message on `in` edge
# `DeltaNode(:out, Marginalisation)` indicates that this rule computes a message out from the edge named `out` and it respects the marginalisation constraint
# `m_in::UnivariateGaussianDistributionsFamily` says that rule expects a Gaussian message (m_) prefix on the edge named `in` 
# `meta::DeltaNodeMeta{F, I, Unscented}` says that rule executes only if approximation method set to `Unscented`
@rule DeltaNode(:out, Marginalisation) (m_in::UnivariateGaussianDistributionsFamily, meta::DeltaNodeMeta{F, I, Unscented}) where {F, I} = begin 
    # call to `ForneyLab.ruleSPDeltaUTOutNG`
    # note from Dmitry: please double check my code :)
    # note from Dmitry 2: Consider trying other methods from ForneyLab
    message = ForneyLab.ruleSPDeltaUTOutNG(
        meta.fn,
        nothing,
        ForneyLab.Message(ForneyLab.Univariate, ForneyLab.Gaussian{ForneyLab.Moments}, m=mean(m_in), v=var(m_in));
        alpha = meta.approximation.alpha
    )
    m, v = ForneyLab.unsafeMeanCov(message.dist)
    return NormalMeanVariance(m, v) 
end

# Rule for outbound message on `in` edge given inbound message on `out` edge
# `DeltaNode(:in, Marginalisation)` indicates that this rule computes a message out from the edge named `in` and it respects the marginalisation constraint
# `m_out::UnivariateGaussianDistributionsFamily` says that rule expects a Gaussian message (m_) prefix on the edge named `out` 
# `m_in::UnivariateGaussianDistributionsFamily` says that rule expects a Gaussian message (m_) prefix on the edge named `out` (remember `default_functional_dependencies_pipeline?`)
# `meta::DeltaNodeMeta{F, Nothing, Unscented}` says that rule executes only if inverse function set to `nothing` and approximation method set to `Unscented`
#  Small quiz from Dmitry: what is the difference between `nothing` and `Nothing`, note that I use capital N in rule specification
@rule DeltaNode(:in, Marginalisation) (m_out::UnivariateGaussianDistributionsFamily, m_in::UnivariateGaussianDistributionsFamily, meta::DeltaNodeMeta{F, Nothing, Unscented}) where {F} = begin     
    # call to `ForneyLab.ruleSPDeltaUTIn1GG` with unknown inverse
    # note from Dmitry: please double check my code :)
    # note from Dmitry 2: Consider implementing the same method from ForneyLab but with known inverse function
    message = ForneyLab.ruleSPDeltaUTIn1GG(
        meta.fn,
        ForneyLab.Message(ForneyLab.Univariate, ForneyLab.Gaussian{ForneyLab.Moments}, m=mean(m_out), v=var(m_out)),
        ForneyLab.Message(ForneyLab.Univariate, ForneyLab.Gaussian{ForneyLab.Moments}, m=mean(m_in), v=var(m_in));
        alpha = meta.approximation.alpha
    )
    m, v = ForneyLab.unsafeMeanCov(message.dist)
    return NormalMeanVariance(m, v) 
end

@rule DeltaNode(:in, Marginalisation) (m_out::MultivariateGaussianDistributionsFamily, m_in::MultivariateGaussianDistributionsFamily, meta::DeltaNodeMeta{F, Nothing, Unscented}) where {F} = begin
    return MvGaussianMeanCovariance(mean(m_out))
end

@rule DeltaNode(:in, Marginalisation) (m_in::MultivariateGaussianDistributionsFamily, meta::DeltaNodeMeta{F, Nothing, Unscented}) where {F} = begin
    return MvGaussianMeanCovariance(mean(m_in))
end

# # The model is oversimplified for demonstration purposes
# @model function nonlinear_estimation(n)
    
#     m  ~ NormalMeanVariance(0.0, 100.0)
#     tm ~ DeltaNode(m)

#     # See `@meta` macro below, but this specification is also possible if someone wants 
#     # tm ~ DeltaNode(m) where { meta = ... }
    
#     y  = datavar(Float64, n)
    
#     for i in 1:n
#         y[i] ~ NormalMeanPrecision(tm, 10.0)
#     end
    
# end

# @meta function nmeta(fn, fn_inv, approximation)
#     # This syntax means that we want to assign RHS meta object (`DeltaNodeMeta`)
#     # to the `DeltaNode` node with `m` and `tm` variables connected to it
#     DeltaNode(m, tm) -> DeltaNodeMeta(fn, fn_inv, approximation)
# end

# # Remember that `Unscented` works decently only for "easy" functions
# # Its considered to be a poor approximation method for non-linear transforms (especially with default parameters)
# nonlinear_fn(x) = 2x - 4

# seed = 123
# rng  = MersenneTwister(seed)

# niters = 15 # Number of VMP iterations

# @show n  = 200 # Number of IID samples
# @show m  = 0.2
# @show tm = nonlinear_fn(m)

# # Create some dummy data for demonstration purposes
# data = rand(rng, NormalMeanPrecision(tm, 10.0), n);

# # call ?inference for more info on this function 
# result = inference(
#     model = Model(nonlinear_estimation, n),
#     meta =  nmeta(nonlinear_fn, nothing, Unscented()),
#     data = (y = data, ), 
#     returnvars = (m = KeepLast(), tm = KeepLast(), ),
#     showprogress = true
# )

# mposterior = result.posteriors[:m]
# tmposterior = result.posteriors[:tm]

# # Compare real values and estimated ones
# @show m, mean_var(mposterior)
# @show tm, mean_var(tmposterior);

# using Plots, StatsPlots

# estimated = Normal(mean_std(mposterior)...)

# plot(estimated, title="Posterior for 'm'", label = "Estimated", legend = :bottomright, fill = true, fillopacity = 0.2, xlim = (-3, 3), ylim = (0, 2))
# vline!([ m ], label = "Real value of 'm'")



# println("test :D")
