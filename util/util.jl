using LinearAlgebra


function sqrtm(M::AbstractMatrix)
    "Square root of matrix"

    if size(M) == (2,2)
        "https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix"

        A,C,B,D = M

        # Determinant
        δ = A*D - B*C
        s = sqrt(δ)

        # Trace
        τ = A+D
        t = sqrt(τ + 2s)

        return 1/t*(M+s*Matrix{eltype(M)}(I,2,2))
    else
        "Babylonian method"

        Xk = Matrix{eltype(M)}(I,size(M))
        Xm = zeros(eltype(M), size(M))

        while sum(abs.(Xk[:] .- Xm[:])) > 1e-3
            Xm = Xk
            Xk = (Xm + M/Xm)/2.0
        end
        return Xk
    end
end

function proj2psd!(S::AbstractMatrix)
    L,V = eigen(S)
    S = V*diagm(max.(1e-8,L))*V'
    return (S+S')/2
end

function sigma_points(m::AbstractVector, P::AbstractMatrix; α=1e-3, κ=0.0)
    
    # Number of sigma points depends on dimensionality
    N = size(P,2)
    
    # Compute scaling parameter
    λ = α^2*(N+κ)-N
    
    # Square root of covariance matrix through Babylonian method
    sP = sqrtm(P)
    
    # Preallocate
    sigma = Matrix(undef,N,2N+1)
    
    # First point is mean
    sigma[:,1] = m
    
    # Positive
    for n = 1:N
        sigma[:,1+n] = m + sqrt(N+λ)*sP[:,n]
    end
            
    # Negative
    for n = 1:N
        sigma[:,1+N+n] = m - sqrt(N+λ)*sP[:,n]
    end
    
    return sigma
end

function ut_weights(; α=1e-3, β=2.0, κ=0.0, N=1)
    
    # Compute scaling parameter
    λ = α^2*(N+κ)-N
    
    # Preallocate
    Wm = Vector(undef, 2N+1)
    Wc = Vector(undef, 2N+1)
    
    # Zero-order weights
    Wm[1] = λ/(N+λ)
    Wc[1] = λ/(N+λ) + (1-α^2+β)
    
    for n = 2:(2N+1)
        Wm[n] = 1/(2(N+λ))
        Wc[n] = 1/(2(N+λ))
    end
    return Wm,Wc
end

function UT(m::AbstractVector, P::AbstractMatrix, g; Q=nothing, D=1, α=1e-3, β=2.0, κ=0.0)
    "Algorithm 5.12 in 'Bayesian filtering & smoothing'"
    
    # Number of sigma points depends on dimensionality
    N = size(P,2)
    
    # Compute constant weigths
    Wm, Wc = ut_weights(α=α, β=β, κ=κ, N=N)
    
    # Form sigma points
    sp = sigma_points(m,P, α=α, κ=κ)
    
    # Propagate sigma points through non-linearity
    if D == 1
        
        # y = Vector{Real}(undef, 2N+1)
        y = zeros(eltype(m), 2N+1)
        for i in 1:(2N+1)
            y[i] = g(sp[:,i])
        end

        # Compute moments of approximated distribution
        μ = y'*Wm
        Σ = Wc[1]*(y[1] - μ)*(y[1] - μ)'
        C = Wc[1]*(sp[:,1] - m)*(y[1] - μ)'
        for i = 2:2N+1
            Σ += Wc[i]*(y[i] - μ)*(y[i] - μ)'
            C += Wc[i]*(sp[:,i] - m)*(y[i] - μ)'
        end
        
        # # Compute moments of approximated distribution
        # μ = dot(y,Wm)
        # Σ = sum([Wc[i]*(y[i] .- μ)*(y[i] .- μ)' for i in 1:(2N+1)])
        # C = sum([Wc[i]*(sp[:,i] .- m)*(y[i] .- μ)' for i in 1:(2N+1)])
    else
        
        y = Matrix(undef, D,2N+1)
        for i in 1:(2N+1)
            y[:,i] = g(sp[:,i])
        end
        
        # Compute moments of approximated distribution
        μ = y*Wm
        Σ = zeros(eltype(m), D,D)
        C = zeros(eltype(m), N,D)
        for i = 1:2N+1
            Σ += Wc[i]*(y[:,i] - μ)*(y[:,i] - μ)'
            C += Wc[i]*(sp[:,i] - m)*(y[:,i] - μ)'
        end
    end
    
    if Q !== nothing; Σ += Q; end
    return μ,Σ,C
end