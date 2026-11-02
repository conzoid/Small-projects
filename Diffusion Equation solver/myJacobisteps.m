function out = myJacobisteps(A, B, n)
    len = length(B);
    sol = zeros(1, len);

    for k = 1:n
        temp_out = sol;
        for i = 1:len 
            temp_coeff = A(i, :);
            temp_coeff(i) = [];
            temp_sol = sol;
            temp_sol(i) = [];
            temp_out(i) = (B(i) - sum(temp_sol.*temp_coeff))/A(i,i);
        end
        sol = temp_out;
    end
    out = sol;
end