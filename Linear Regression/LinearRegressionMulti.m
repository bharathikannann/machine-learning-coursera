
%% Clear and Close Figures
clear ; close all; clc
% ====================== Normalization ======================

function [X_norm, mu, sigma] = featureNormalize(X)
  X_norm = X;
  mu = zeros(1, size(X, 2));
  sigma = zeros(1, size(X, 2));
  sigma=std(X);
  mu=mean(X);
  X_norm=(X-mu)./sigma;
end

% ====================== Cost Function =====================

function J = computeCostMulti(X, y, theta)
  m = length(y); % number of training examples
  J = 0;
  J=(1/(2*m))*sum(((X*theta)-y).^2);
end

% ====================== GradientDescentMulti ======================

function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
  m = length(y); % number of training examples
  J_history = zeros(num_iters, 1);

  for iter = 1:num_iters
    error=(X*theta)-y;
    theta=theta-((alpha/m)*(error'*X)');
    J_history(iter) = computeCostMulti(X, y, theta);
  end

end

% ====================== Normal Equations ======================

function [theta] = normalEqn(X, y)
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.
  theta = zeros(size(X, 2), 1);
  theta = pinv(X'*X) * X' * y;
end


%% ================ Part 1: Feature Normalization ================

  fprintf('Loading data ...\n');

  %% Load Data
  data = load('ex1data2.txt');
  X = data(:, 1:2);
  y = data(:, 3);
  m = length(y);

  % Print out some data points
  fprintf('First 10 examples from the dataset: \n');
  fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

  fprintf('Program paused. Press enter to continue.\n');
  pause;

  % Scale features and set them to zero mean
  fprintf('Normalizing Features ...\n');

  [X mu sigma] = featureNormalize(X);

  % Add intercept term to X
  X = [ones(m, 1) X];


%% ================ Part 2: Gradient Descent ================

  fprintf('Running gradient descent ...\n');

  % Choose some alpha value
  alpha = 0.01;
  num_iters = 400;

  % Init Theta and Run Gradient Descent 
  theta = zeros(3, 1);
  [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

  % Plot the convergence graph
  figure;
  plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
  xlabel('Number of iterations');
  ylabel('Cost J');

  % Display gradient descent's result
  fprintf('Theta computed from gradient descent: \n');
  fprintf(' %f \n', theta);
  fprintf('\n');

  % Estimate the price of a 1650 sq-ft, 3 br house
  price = 100; %change this

  fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
           '(using gradient descent):\n $%f\n'], price);

  fprintf('Program paused. Press enter to continue.\n');
  pause;

%% ================ Part 3: Normal Equations ================

  fprintf('Solving with normal equations...\n');

  %% Load Data
  data = csvread('ex1data2.txt');
  X = data(:, 1:2);
  y = data(:, 3);
  m = length(y);

  % Add intercept term to X
  X = [ones(m, 1) X];

  % Calculate the parameters from the normal equation
  theta = normalEqn(X, y);

  % Display normal equation's result
  fprintf('Theta computed from the normal equations: \n');
  fprintf(' %f \n', theta);
  fprintf('\n');

  price = 100; %change this

  fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
           '(using normal equations):\n $%f\n'], price);

