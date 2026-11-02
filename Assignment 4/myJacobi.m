function [out, iteration] = myJacobi(A, B)
%function uses Jacobi iteration with to solve a linear set of equations
%in the form of Ax=b. Pass the matrix A and B to output the solution matrix
%and the number of iteration.

    len = length(B);
    sol = zeros(1, len);
    error = 100;
    iteration = 0;

    while error>1e-5
        temp_out = sol;
        for i = 1:len 
            temp_coeff = A(i, :);
            temp_coeff(i) = [];
            temp_sol = sol;
            temp_sol(i) = [];
            temp_out(i) = (B(i) - sum(temp_sol.*temp_coeff))/A(i,i);
        end
        error = sum(abs(temp_out - sol));
        sol = temp_out;
        iteration = iteration+1;
    end
    out = sol;
end