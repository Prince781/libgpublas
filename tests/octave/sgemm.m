% computes GFLOPS

N = 2;
trials = 10;
while N <= 16384
    elapsedTime = 0;
    for i = 1:(1 + trials)
            A = single(rand(N,N));
            B = single(rand(N,N));
            start = clock();
            C = A * B;
            if (i > 1)
                elapsedTime += etime(clock(), start);
            endif
    endfor
    gFlops = 2*N*N*N/(elapsedTime * 1e+9);
    printf('OCTAVE SGEMM[n=%d]: elapsed=%f s GFLOPS=%f\n', 
        N, elapsedTime / trials, gFlops);
    N = N * 2;
end
