@testset "Adjust input data" begin 
    # Vectors
    K = 10
    u = rand(K)
    @test size(adjust_input(u, 1, K)) == (1, K)
    @test size(adjust_input(u', 1, K)) == (1, K)

    # Matrices
    m = 3
    U = rand(m, K)
    @test size(adjust_input(U, m, K)) == (m, K)
    @test size(adjust_input(U', m, K)) == (m, K)
end
