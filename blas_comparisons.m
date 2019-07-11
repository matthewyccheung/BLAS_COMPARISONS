clc
close all
clear all

funcs = {'matmul_triple', 'matmul_dot', 'matmul_saxpy' ...
            'matmul_matrix_vector', 'matmul_outer'};

mags = round(linspace(100, 500, 10));
ts = zeros(length(mags), length(funcs));


for i = 1:length(mags)
    m = mags(i); n = mags(i); p = mags(i);
    A = rand(m, p);
    B = rand(p, n);
    for j = 1:length(funcs)
        [C, t] = eval([char(funcs(j)), '(A, B)']);
        ts(i, j) = t;
    end
end

% Plot time required
figure; hold on;
for i = 1:size(ts, 2)
    plot(mags, ts(:, i));
end
legend('triple', 'dot (BLAS-1)', 'saxpy (BLAS-1)', 'matrix_vector (BLAS-2)', 'outer (BLAS-2)');
title('Run Time Comparison of 5 Different Matrix Multiplication Methods');
ylabel('Run Time (s)'); xlabel('size of matrix (nxn)')
hold off;


ttemp = ts;

% % Plot GFLOPS/s
% for i = 1:size(mags, 2)
%     ts(i, :) = 2.*(mags(i).^3)./(1e6)./(ts(i, :));
% end

ts(i, :) = 2.*(mags(i).^3)./(1e9)./(ts(i, :));
ts(i, :) = 2.*(mags(i).^3)./(1e9)./(ts(i, :));
ts(i, :) = 2.*(mags(i).^3)./(1e9)./(ts(i, :));
ts(i, :) = 2.*(mags(i).^3)./(1e9)./(ts(i, :));
ts(i, :) = 2.*(mags(i).^3)./(1e9)./(ts(i, :));


figure; hold on;
for i = 1:size(ts, 2)
    plot(mags, ts(:, i));
end
legend('triple', 'dot', 'saxpy', 'matrixvector', 'outer');
title('GFLOPS/s Comparison of 5 Different Matrix Multiplication Methods');
hold off;


varnames = {'array_size_nxn', 'matmul_triple_run_time_s', ...
            'matmul_dot_run_time_s', 'matmul_saxpy_run_time_s', ...
            'matmul_vector_run_time_s', 'matmul_outer_run_time_s'};
Tdata = [mags', ts]; T = array2table(Tdata);
T.Properties.VariableNames = varnames;




function [C, t] = matmul_triple(A, B)
    % Method 1: Triple Loop
    [m, n, p] = size2mat(A, B);
    C = zeros(m, n);
    tstart = tic;
    for i=1:m
        for j=1:n
            C(i,j) = 0;
            for k=1:p
                C(i,j) = C(i,j) + A(i,k)*B(k,j);
            end
        end
    end
    t = toc(tstart);
end

function [C, t] = matmul_dot(A, B) 
    % Method 2: Dot Product (BLAS 1)
    [m, n, p] = size2mat(A, B);
    C = zeros(m, n);
    tstart = tic;
    for i = 1:m
        for j = 1:n
            C(i, j) = A(i,:)*B(:, j);
        end
    end
    t = toc(tstart);
end

function [C, t] = matmul_saxpy(A, B)
    % Method 3: saxpy (BLAS 1)
    [m, n, p] = size2mat(A, B);
    C = zeros(m, n);
    tstart = tic;
    for j = 1:n
        for k = 1:p
            C(:, j) = C(:, j) + A(:, k)*B(k, j);
        end
    end
    t = toc(tstart);
end

function [C, t] = matmul_matrix_vector(A, B)
    % Method 4: Matrix Vector Multiply (BLAS 2)
    [m, n, p] = size2mat(A, B);
    C = zeros(m, n);
    tstart = tic;
    for j = 1:n
        C(:, j) = A*B(:, j);
    end
    t = toc(tstart);
end

function [C, t] = matmul_outer(A, B)
    % Method 5: Outer Product (BLAS 2)
    [m, n, p] = size2mat(A, B);
    C = zeros(m, n);
    tstart = tic;
    for k = 1:p
        C(:,:) = C(:,:) + A(:, k)*B(k, :);
    end
    t = toc(tstart);
end

function [m, n, p] = size2mat(A, B)
    [m, p] = size(A);
    [p, n] = size(B);
end