function [bounds, output_flag] = measureOperator(O, inferred_rho, povms)
% Finds lower and upper bounds of expectation value for inferred state.
%
% This function requires YALMIP and SeDuMi are installed. To install, see:
%  
% YALMIP: https://yalmip.github.io/
%
% SeDuMi: http://sedumi.ie.lehigh.edu
%
% 
% Inputs:
% -------
% O: observable to find expectation value
% inferred_rho: inferred state from joint state and measurement tomography
% povms: povm elements that were measured
%
% Outputs:
% --------
% bounds: tuple of lower and upper expectation value
% output_flag: integer that describes stopping condtion
%              (see https://yalmip.github.io/command/yalmiperror/)


dim = size(povms, 1);        % dimension of density matrix
num_povms = size(povms, 3);  % number of povm elements (over all povms)

rho = sdpvar(dim, dim, 'hermitian', 'complex');
bounds = zeros(1,2);

% Constraints for both bounds
C = [];

% rho is positive and trace 1
C = [rho >= 0, trace(rho) == 1]; 

% outcome of rho has same probability as each outcome from inferred_rho
for k = 1:num_povms
    C = [C, trace((rho-inferred_rho)*povms(:,:,k)) == 0];
end

% Set some options for YALMIP and solver
options = sdpsettings('verbose',0,'solver','sedumi');

% yalmip always minimizes objective
Objective = trace(O*rho);

% minimum expectation value
diag_lower = optimize(C, Objective, options);
output_flag(1) = diag_lower.problem;
bounds(1,1) = trace(O*value(rho));

% maximum expectation value
diag_upper = optimize(C, -Objective, options);
output_flag(2) = diag_upper.problem;
bounds(1,2) = trace(O*value(rho));

end
