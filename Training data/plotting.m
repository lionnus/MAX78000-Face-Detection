%% QAT effect
% Read the first CSV file with original variable names
data1 = readtable('widerfaceonet_v4_qatafter140epochs_valloss.csv', 'VariableNamingRule', 'preserve');

% Extract columns using original column headers
step1 = data1.Step;
value1 = data1.Value;

% Read the second CSV file with original variable names
data2 = readtable('widerfaceonet_v4_noqat_210epochs_valloss.csv', 'VariableNamingRule', 'preserve');

% Extract columns using original column headers
step2 = data2.Step;
value2 = data2.Value;

% Apply smoothing to the values
smoothedValue1 = smoothdata(value1, 'gaussian', 5); % Adjust the smoothing parameters as desired
smoothedValue2 = smoothdata(value2, 'gaussian', 5); % Adjust the smoothing parameters as desired

% Plot the values against the step
plot(step1, smoothedValue1, 'r-', 'LineWidth', 1.5);
hold on;
plot(step2, smoothedValue2, 'k--', 'LineWidth', 2);
hold off;

% Set the y-axis and x-axis limits
ylim([0.02 0.1]);
xlim([0 210]);

% Add labels and title
xlabel('Epoch');
ylabel('MSE/Loss');
title('Validation loss of ONet with and without QAT');

% Add legend
legend('QAT after 140 epochs','No QAT');
% Save the plot as a PDF file
saveas(gcf, 'plot_onet_qat-vs-noqat.pdf');
print('plot_onet_qat-vs-noqat.eps', '-depsc', '-vector');
%% ONet vs RNet
clear; close all;
% Read the first CSV file with original variable names
data1 = readtable('widerfaceonet_v4_qatafter140epochs_valloss.csv', 'VariableNamingRule', 'preserve');

% Extract columns using original column headers
step1 = data1.Step;
value1 = data1.Value;

% Read the second CSV file with original variable names
data2 = readtable('widerfacernet_v4_qatafter140epochs_valloss.csv', 'VariableNamingRule', 'preserve');

% Extract columns using original column headers
step2 = data2.Step;
value2 = data2.Value;

% Apply smoothing to the values
smoothedValue1 = smoothdata(value1, 'gaussian', 5); % Adjust the smoothing parameters as desired
smoothedValue2 = smoothdata(value2, 'gaussian', 5); % Adjust the smoothing parameters as desired

% Plot the values against the step
plot(step1, smoothedValue1, 'r-', 'LineWidth', 1.5);
hold on;
plot(step2, smoothedValue2,'Color', [0, 140, 217]/255, 'LineStyle', '-', 'LineWidth', 1.5);
hold off;

% Set the y-axis and x-axis limits
ylim([0.018 0.14]);
xlim([0 240]);

% Add labels and title
xlabel('Epoch');
ylabel('MSE/Loss');
title('Comparison of ONet and RNet Validation loss');

% Add legend
legend('ONet', 'RNet');
% Save the plot as a PDF file
saveas(gcf, 'plot_onet-vs-rnet.pdf');
print('plot_onet-vs-rnet.eps', '-depsc', '-vector');
