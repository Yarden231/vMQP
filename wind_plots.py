import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter



def rose_plot(ax, angles, bins=32, density=None, offset=np.pi/2, lab_unit="degrees",
              start_zero=False, fontsize=14, **param_dict):
    """
    Plot polar histogram of angles on ax. ax must have been created using
    subplot_kw=dict(projection='polar'). Angles are expected in radians.
    """
    if start_zero:
        if bins % 2:
            bins += 1
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    count, bin = np.histogram(angles, bins=bins)
    widths = np.diff(bin)

    if density is None or density is True:
        area = count / angles.size
        radius = (area / np.pi)**.5
    else:
        radius = count

    ax.bar(bin[:-1], radius, zorder=1, align='edge', width=widths,
           edgecolor='C0', fill=False, linewidth=1)
    ax.set_theta_offset(offset)
    ax.set_yticks([])

    if lab_unit == "radians":
        label = ['$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$',
                 r'$\pi$', r'$5\pi/4$', r'$3\pi/2$', r'$7\pi/4$']
        ax.set_xticklabels(label, fontsize=fontsize)

def plot_selected_predictions(testPredVonMises_10th_row, column_indices=[4, 17, 30, 43], 
                            bins=32, output_path='angles_rose_trace.pdf'):
    """
    Create combined line and rose plots for selected predictions.
    """
    num_selected_plots = len(column_indices)
    
    fig, axs = plt.subplots(2, num_selected_plots, figsize=(num_selected_plots * 4, 7))
    fig.subplots_adjust(hspace=0.5, wspace=0.4)

    # First row: Line plots
    for i, p in enumerate(column_indices):
        axs[0, i].plot(testPredVonMises_10th_row[p, :])
        axs[0, i].xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
        axs[0, i].tick_params(axis='both', which='major', labelsize=14)

    # Second row: Rose plots
    for i, p in enumerate(column_indices):
        ax_polar = fig.add_subplot(2, num_selected_plots, num_selected_plots + i + 1, 
                                 projection='polar')
        rose_plot(ax_polar, testPredVonMises_10th_row[p, :], lab_unit="radians", 
                 bins=bins, fontsize=14)
        
        # Hide the original axis
        axs[1, i].set_frame_on(False)
        axs[1, i].set_yticklabels([])
        axs[1, i].set_xticklabels([])
        axs[1, i].tick_params(left=False)
        axs[1, i].tick_params(bottom=False)
        axs[1, i].spines['bottom'].set_visible(False)

    plt.tight_layout()
    #plt.savefig(output_path, format='pdf', bbox_inches='tight')
    return fig, axs

def plot_parameter_traces(samples, i, N_burnin_trace=0, N_burnin_hist=0, 
                         bins=30, fontsize1=18, output_path="von_mises_traces_histograms.pdf"):
    """
    Plot trace plots and histograms for model parameters with LaTeX labels and formatted axes.
    """
    fig, axs = plt.subplots(2, 4, figsize=(20, 8))
    fig.subplots_adjust(hspace=0.5, wspace=0.4)

    # Parameters configuration
    params = {
        'l1': {
            'title': r'$\mathbf{\sigma^2}$',
            'position': (0, 0)
        },
        'l2': {
            'title': r'$\mathbf{l^2}$',
            'position': (0, 1)
        },
        'nu': {
            'title': r'$\mathbf{\nu}$',
            'position': (0, 2)
        },
        'kappa': {
            'title': r'$\mathbf{\kappa}$',
            'position': (0, 3)
        }
    }

    for param_name, config in params.items():
        row, col = config['position']
        param_values = getattr(samples, f"{param_name}s")
        
        # Trace plot
        axes = axs[0, col]
        axes.plot(param_values[N_burnin_trace:i])
        axes.set_title(config['title'], fontsize=fontsize1)
        axes.axhline(np.mean(param_values[N_burnin_trace:i]), color='b', linewidth=2)
        axes.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))

        # Histogram
        axes = axs[1, col]
        axes.hist(param_values[N_burnin_hist:i], alpha=0.75, bins=bins)
        axes.axvline(np.mean(param_values[N_burnin_hist:i]), color='b', linewidth=2)
        axes.set_title(config['title'], fontsize=fontsize1)
        axes.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, format='pdf')
    return fig, axs

def plot_train_test_split(combined_xs, ys_all, ind_obs, ind_un, map_path="germanyMap.png", 
                         alpha=0.2):
    """
    Plot the train-test split of wind direction data on a map of Germany.
    """
    img = plt.imread(map_path)
    bins = 50
    scale = 30
    headlength2 = 5
    headwidth2 = 4
    width2 = 0.002
    headaxislength2 = headlength2 - 1

    fig, ax = plt.subplots(figsize=(8, 11))

    # Define colors
    Teal = (0.0, 0.2, 0.2)
    dark_red = (0.6, 0.0, 0.0)

    # Plot wind directions
    ax.quiver(combined_xs[:, 0], combined_xs[:, 1], 
             np.cos(ys_all), -np.sin(ys_all), 
             color=Teal, alpha=0.6, 
             headaxislength=headaxislength2, 
             headwidth=headwidth2, 
             headlength=headlength2, 
             width=width2, 
             scale=scale)
    
    ax.quiver(combined_xs[ind_un, 0], combined_xs[ind_un, 1], 
             np.cos(ys_all[ind_un]), -np.sin(ys_all[ind_un]), 
             color=dark_red, alpha=0.9, 
             headaxislength=headaxislength2, 
             headwidth=headwidth2, 
             headlength=headlength2, 
             width=width2, 
             scale=scale)

    # Plot map and set labels
    ax.imshow(img, extent=[5.5, 15, 46.5, 55.5], alpha=alpha)
    ax.set_title("Division of Data into Train and Test Sets", 
                fontsize=18, pad=10)
    ax.title.set_position([.5, 1.9])
    ax.legend(['Train', 'Test'], loc='upper left', 
             bbox_to_anchor=(0.9, 1))
    ax.set_xlabel('Longitude', fontsize=14)
    ax.set_ylabel('Latitude', fontsize=14)
    ax.set_xlim(5.5, 15)
    ax.set_ylim(46.5, 55.5)
    plt.tight_layout()
    return fig, ax

def plot_von_mises_predictions(testPredVonMises_10th_row, test_ys, CRPS_vec_VM, 
                             num_rows=4, num_cols=13, bins=20):
    """
    Plot predictions from Von Mises model in different formats.
    """
    # Trace plots
    fig1, axs1 = plt.subplots(num_rows, num_cols, 
                             figsize=(num_cols*4, num_rows*4))
    axs1_flat = axs1.flatten()

    # Histogram plots
    fig2, axs2 = plt.subplots(num_rows, num_cols, 
                             figsize=(num_cols*4, num_rows*4))
    axs2_flat = axs2.flatten()

    # Rose plots
    fig3, axs3 = plt.subplots(num_rows, num_cols, 
                             figsize=(num_cols*4, num_rows*4), 
                             subplot_kw=dict(projection='polar'))
    axs3_flat = axs3.flatten()

    for p in tqdm(range(len(axs1_flat))):
        # Trace plot
        axs1_flat[p].plot(testPredVonMises_10th_row[p,:])
        axs1_flat[p].set_xlabel('Iterations')
        axs1_flat[p].set_title(f'CRPS = {CRPS_vec_VM[p]}')
        axs1_flat[p].axhline(test_ys[p], color='black', linewidth=1)

        # Histogram plot
        axs2_flat[p].hist(testPredVonMises_10th_row[p,:], bins=bins)
        axs2_flat[p].set_xlabel('Iterations')
        axs2_flat[p].set_title(f'CRPS = {CRPS_vec_VM[p]}')
        axs2_flat[p].axvline(test_ys[p], color='black', linewidth=1)

        # Rose plot
        rose_plot(axs3_flat[p], testPredVonMises_10th_row[p,:], 
                 lab_unit="radians", bins=bins)
        axs3_flat[p].quiver(0, 0, np.sin(test_ys[p]), np.cos(test_ys[p]), 
                           color='black', scale=1, scale_units='xy', 
                           angles='xy')
        axs3_flat[p].set_title(f'CRPS = {CRPS_vec_VM[p]}')

    for fig in [fig1, fig2, fig3]:
        fig.tight_layout()
    
    return fig1, fig2, fig3

def plot_wind_directions(combined_xs, ys_all, ind_obs, ind_un, testPredVonMises, 
                        map_path="germanyMap.png", alpha_img=0.2):
    """
    Plot observed and predicted wind directions on a map.
    """
    img = plt.imread(map_path)
    scale = 25
    headlength2 = 5
    headwidth2 = 4
    width2 = 0.002
    headaxislength2 = headlength2 - 1

    # Calculate circular mean of predictions
    y_circmean_VM = stats.circmean(testPredVonMises, high=2*np.pi, low=0, axis=0)
    y_circmean_VM = -np.pi + y_circmean_VM
    # Convert directions for visualization
    observed_directions = -np.cos(ys_all) - 1j * np.sin(ys_all)
    predicted_directions_VM = -np.cos(y_circmean_VM) - 1j * np.sin(y_circmean_VM)

    # Define colors
    dteal = (0.0, 0.2, 0.2)
    dark_red = (0.6, 0.0, 0.0)
    lteal = (0.1, 0.5, 0.5)

    fig, ax = plt.subplots(figsize=(10, 15))

    # Plot directions
    ax.quiver(combined_xs[ind_obs, 0], combined_xs[ind_obs, 1], 
             observed_directions[ind_obs].real, 
             observed_directions[ind_obs].imag, 
             color=lteal, headaxislength=headaxislength2, 
             headwidth=headwidth2, headlength=headlength2, 
             width=width2, scale=scale)

    ax.quiver(combined_xs[ind_un, 0], combined_xs[ind_un, 1], 
             observed_directions[ind_un].real, 
             observed_directions[ind_un].imag, 
             color=dteal, headaxislength=headaxislength2, 
             headwidth=headwidth2, headlength=headlength2, 
             width=width2, scale=scale)

    ax.quiver(combined_xs[ind_un, 0], combined_xs[ind_un, 1], 
             predicted_directions_VM.real, predicted_directions_VM.imag, 
             color=dark_red, headaxislength=headaxislength2, 
             headwidth=headwidth2, width=width2, 
             headlength=headlength2, scale=scale)

    # Plot map and set labels
    ax.imshow(img, extent=[5.5, 15, 46.5, 55.5], alpha=alpha_img)
    ax.legend(['Train', 'Test', 'vM Predictions'], 
             loc='upper left', bbox_to_anchor=(0, 1))
    ax.set_xlabel('Longitude', fontsize=16)
    ax.set_ylabel('Latitude', fontsize=16)
    ax.set_xlim(5.5, 15)
    ax.set_ylim(46.5, 55.5)

    return fig, ax

def thousands_formatter(x, pos):
    if x >= 1000:
        return f'{int(x / 1000)}k'
    else:
        return f'{int(x)}'