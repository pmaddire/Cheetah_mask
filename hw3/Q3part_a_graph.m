% Data
alpha = [1e-4, 1e-3, 0.01, 0.1, 1, 10, 100, 1000, 10000]; % Alpha values
percent_error = [8.87, 8.6, 9.53, 9.69, 9.7066, 9.7066, 9.7066, 9.7066, 9.7066]; % Percent error values

% Create log-log plot
figure;
semilogx(alpha, percent_error, '-o', 'LineWidth', 1.5, 'MarkerSize', 6);
grid on;

% Add labels and title
xlabel('Alpha Value (log scale)', 'FontSize', 12);
ylabel('Percent Error (%)', 'FontSize', 12);
title('Logarithmic Plot of Alpha Value vs Percent Error', 'FontSize', 14);

% Enhance appearance
set(gca, 'FontSize', 12);
