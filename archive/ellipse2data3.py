import os
import numpy as np
import matplotlib.pyplot as plt
import math
import argparse
import re
import pandas as pd
from collections import defaultdict
from scipy.signal import butter, filtfilt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize



plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams.update({'font.size': 16})

cmap_f = plt.get_cmap('autumn')
cmap_r = plt.get_cmap('winter')


def butter_lowpass(cutoff, fs, order=5):
    """Design a Butterworth lowpass filter."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff=2.0, fs=30.0, order=5):
    """Apply lowpass filter to data."""
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def natural_sort_key(s):
    """Natural sorting for filenames."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def readEllipse(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    ellipses = []

    for line in lines:
        parts = list(map(float, line.strip().split()))

        class_id = int(parts[0])
        points = np.array(parts[1:]).reshape(-1, 2)
        center = np.mean(points, axis=0)
        centered_points = points - center
        cov = np.cov(centered_points.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        major_axis = 2 * math.sqrt(max(eigenvalues))
        minor_axis = 2 * math.sqrt(min(eigenvalues))
        angle = math.degrees(math.atan2(eigenvectors[1, 0], eigenvectors[0, 0]))

        ellipses.append({
            'class_id': class_id,
            'center': center,
            'major_axis': major_axis,
            'minor_axis': minor_axis,
            'angle': angle
        })

    return ellipses
    
def process_directory(directory_path):
    """Process all .txt files in directory."""
    files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
    files.sort()#key=natural_sort_key)
    
    tracking_data = defaultdict(lambda: {
        'x_pos': [], 'y_pos': [], 'major_axes': [], 'minor_axes': [], 'frames': []
    })
    
    for frame_idx, filename in enumerate(files):
        file_path = os.path.join(directory_path, filename)
        for ellipse in readEllipse(file_path):
            class_id = ellipse['class_id']
            tracking_data[class_id]['x_pos'].append(ellipse['center'][0])
            tracking_data[class_id]['y_pos'].append(ellipse['center'][1])
            tracking_data[class_id]['major_axes'].append(ellipse['major_axis'])
            tracking_data[class_id]['minor_axes'].append(ellipse['minor_axis'])
            tracking_data[class_id]['frames'].append(frame_idx)
    
    return tracking_data


def apply_filters(tracking_data, cutoff=2.0, fs=30.0):
    """Apply lowpass filters to all tracking data."""
    filtered_data = {}
    for class_id, data in tracking_data.items():
        filtered = {
            'x_pos': lowpass_filter(data['x_pos'], cutoff, fs),
            'y_pos': lowpass_filter(data['y_pos'], cutoff, fs),
            'major_axes': lowpass_filter(data['major_axes'], cutoff, fs),
            'minor_axes': lowpass_filter(data['minor_axes'], cutoff, fs),
            'frames': data['frames']
        }
        filtered_data[class_id] = filtered
    return filtered_data

def calculate_differences(filtered_data, absolute=True):
    """Calculate differences between both ellipses."""
    if len(filtered_data) < 2:
        return None
    
    # Get common frame range
    min_frames = min(len(filtered_data[0]['x_pos']), len(filtered_data[1]['x_pos']))
    frames = np.arange(min_frames)
    
    # Calculate differences (Y values are already inverted)
    x_diff = np.array(filtered_data[0]['x_pos'][:min_frames]) - np.array(filtered_data[1]['x_pos'][:min_frames])
    y_diff = np.array(filtered_data[1]['y_pos'][:min_frames]) - np.array(filtered_data[0]['y_pos'][:min_frames])  # Inverted subtraction
    
    euclidean_diff = np.sqrt(x_diff**2 + y_diff**2)

    if absolute:
        x_diff = np.abs(x_diff)
        y_diff = np.abs(y_diff)
    
    # Calculate statistics
    max_x_diff = np.max(x_diff)
    max_y_diff = np.max(y_diff)
    avg_x_diff = np.mean(x_diff)
    avg_y_diff = np.mean(y_diff)
    
    # Calculate percentage variations
    x_diff_pct = ((max_x_diff - avg_x_diff) / avg_x_diff) * 100 if avg_x_diff != 0 else 0
    y_diff_pct = ((max_y_diff - avg_y_diff) / avg_y_diff) * 100 if avg_y_diff != 0 else 0
    
    # Print statistics to console
    print("\n=== Motion ===")
    print(f"Maximum X difference: {max_x_diff:.2f} pixels ({x_diff_pct:.1f}% above average)")
    print(f"Maximum Y difference: {max_y_diff:.2f} pixels ({y_diff_pct:.1f}% above average)")
    print(f"Average X difference: {avg_x_diff:.2f} pixels")
    print(f"Average Y difference: {avg_y_diff:.2f} pixels")
    print("=================================")
    
    return {
        'x_difference': x_diff,
        'y_difference': y_diff,
        'frames': frames,
        'max_x_diff': max_x_diff,
        'max_y_diff': max_y_diff,
        'euclidean_distance': euclidean_diff
    }


def plot_data(tracking_data, filtered_data, show_raw=False, absolute_diff=True):
    """Plot data with specified options."""
    fig = plt.figure(figsize=(18, 12))  # Adjusted figure size
    
    # Define color scheme
    ellipse_colors = ['turquoise', 'red']
    # ellipse_cmaps = ['OrRd', 'YlGnBu']
    ellipse_cmaps = ['autumn', 'winter']
    diff_colors = ['green', 'orange']
    raw_alpha = 0.3 if show_raw else 0

    frame_count = max(len(data['frames']) for data in tracking_data.values())
    print(f"Total frames processed: {frame_count}")
    frames_norm = np.linspace(0, 1, frame_count)

    # Map colormaps
    # frames_f_norm = plt.Normalize(vmin=min(filtered_data[0]['frames']), vmax=max(filtered_data[0]['frames']))
    # frames_r_norm = plt.Normalize(vmin=min(filtered_data[1]['frames']), vmax=max(filtered_data[1]['frames']))
    colors_f = cmap_f(frames_norm)
    colors_r = cmap_r(frames_norm)
    ellipse_cmaps = [colors_f, colors_r]
    
    # Calculate differences
    diff_metrics = calculate_differences(filtered_data, absolute_diff)
    
    # Plot X Position
    ax1 = plt.subplot(3, 3, 1)
    for class_id, data in tracking_data.items():
        if class_id == 0:
            class_name = 'Front'
        elif class_id == 1:
            class_name = 'Rear'

        if show_raw:
            ax1.plot(data['frames'], data['x_pos'], '-', color=ellipse_colors[class_id], 
                    alpha=raw_alpha, label=f'{class_name} wheel raw')
        ax1.plot(filtered_data[class_id]['frames'], filtered_data[class_id]['x_pos'], '-', 
                color=ellipse_colors[class_id], linewidth=2, label=f'{class_name} wheel filtered')
    ax1.set_title('$x$ Position', fontweight='bold')
    ax1.set_ylabel('$x$ position (pixels)')#, fontsize = 16)
    ax1.legend()
    ax1.grid()
    
    # Plot Y Position (inverted for user-friendly view)
    ax2 = plt.subplot(3, 3, 2)
    for class_id, data in tracking_data.items():
        if show_raw:
            ax2.plot(data['frames'], data['y_pos'], '-', color=ellipse_colors[class_id], 
                    alpha=raw_alpha, label=f'{class_name} wheel Y raw')
        ax2.plot(filtered_data[class_id]['frames'], filtered_data[class_id]['y_pos'], '-', 
                color=ellipse_colors[class_id], linewidth=2, label=f'{class_name} wheel filtered')
    ax2.set_title('$y$ Position', fontweight='bold')
    ax2.set_ylabel('$y$ position (pixels)')
    ax2.invert_yaxis()
    ax2.grid()
    
    # Plot Major/Minor Axis Ratio
    ax3 = plt.subplot(3, 3, 3)
    for class_id, data in tracking_data.items():

        if class_id == 0:
            class_name = 'Front'
        elif class_id == 1:
            class_name = 'Rear'

        # Calculate ratio
        if show_raw:
            ratio_raw = np.array(data['major_axes']) / np.array(data['minor_axes'])
            ax3.plot(data['frames'], ratio_raw, '-', color=ellipse_colors[class_id], 
                    alpha=raw_alpha, label=f'{class_name} wheel ellipse ratio')
        
        ratio_filtered = np.array(filtered_data[class_id]['major_axes']) / np.array(filtered_data[class_id]['minor_axes'])
        ax3.plot(filtered_data[class_id]['frames'], ratio_filtered, '-', 
                color=ellipse_colors[class_id], linewidth=2, label=f'{class_name} wheel ellipse ratio')
    ax3.set_title('Major/Minor Axis Ratio', fontweight='bold')
    ax3.set_ylabel('Ratio (major/minor)')
    ax3.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax3.legend()
    ax3.grid()
    
    # # Plot Angle
    # ax4 = plt.subplot(2, 3, 6)
    # for class_id, data in tracking_data.items():
    #     if show_raw:
    #         ax4.plot(data['frames'], data['angles'], '-', color=ellipse_colors[class_id], 
    #                 alpha=raw_alpha, label=f'{class_name} wheel')
    #     ax4.plot(filtered_data[class_id]['frames'], filtered_data[class_id]['angles'], '-', 
    #             color=ellipse_colors[class_id], linewidth=2, label=f'{class_name} wheel')
    # ax4.set_title('Angle', fontweight='bold')
    # ax4.set_xlabel('Frame number')
    # ax4.set_ylabel('Angle (degrees)')
    # ax4.grid()
    
    ax4 = plt.subplot(3, 3, 6)
    if diff_metrics:
        ax4.plot(diff_metrics['frames'], diff_metrics['euclidean_distance'], '-', 
                color='purple', label='Euclidean Distance')
        
        diff_type = 'Absolute' if absolute_diff else 'Signed'
        ax4.set_title(f'Euclidean Distance between Ellipses', fontweight='bold')
        ax4.set_xlabel('Frame number')
        ax4.set_ylabel(f'Distance (pixels)')
        ax4.legend()
        ax4.grid()


    # Plot X Differences
    ax5 = plt.subplot(3, 3, 4)
    if diff_metrics:
        ax5.plot(diff_metrics['frames'], diff_metrics['x_difference'], '-', 
                color=diff_colors[0], label='$x$ Difference')
        
        diff_type = 'Absolute' if absolute_diff else 'Signed'
        ax5.set_title(f'$x$ Distance', fontweight='bold')
        ax5.set_xlabel('Frame number')
        ax5.set_ylabel(f'$x$ Distance (pixels)')
        ax5.legend()
        ax5.grid()
        
        if not absolute_diff:
            ax5.axhline(0, color='black', linestyle='--', alpha=0.5)
    else:
        ax5.text(0.5, 0.5, 'Not enough ellipses for difference calculation', 
                ha='center', va='center')
        ax5.set_title('X Differences (Not Available)')
    
    # Plot Y Differences
    ax6 = plt.subplot(3, 3, 5)
    if diff_metrics:
        ax6.plot(diff_metrics['frames'], diff_metrics['y_difference'], '-', 
                color=diff_colors[1], label='Y Difference')
        
        diff_type = 'Absolute' if absolute_diff else 'Signed'
        ax6.set_title(f'$y$ Distance', fontweight='bold')
        ax6.set_xlabel('Frame number')
        ax6.set_ylabel(f'$y$ Distance (pixels)')
        ax6.legend()
        ax6.grid()
        
        if not absolute_diff:
            ax6.axhline(0, color='black', linestyle='--', alpha=0.5)
    else:
        ax6.text(0.5, 0.5, 'Not enough ellipses for difference calculation', 
                ha='center', va='center')
        ax6.set_title('Y Differences (Not Available)')

    ax7 = plt.subplot(3, 3, 7)
    if diff_metrics:
        ax7.scatter(diff_metrics['x_difference'], diff_metrics['y_difference'])
        ax7.set_title('Centres Differences', fontweight='bold')
        ax7.set_xlabel('X Difference (pixels)')
        ax7.set_ylabel('Y Difference (pixels)')
        ax7.grid()
    

    ax8 = plt.subplot(3, 3, 8)
    for class_id, data in tracking_data.items():
        if class_id == 0:
            class_name = 'Front'
            class_cmap = 'autumn'
        elif class_id == 1:
            class_name = 'Rear'
            class_cmap = 'winter'

        if show_raw:
            ax8.plot(data['x_pos'], data['y_pos'], '-', color=ellipse_colors[class_id], 
                    alpha=raw_alpha, label=f'{class_name} wheel positionraw')
            
        
        x = np.array(filtered_data[class_id]['x_pos'])
        y = np.array(filtered_data[class_id]['y_pos'])
        frames = np.array(filtered_data[class_id]['frames'])

        # Create segments for gradient line
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Normalize frame values for colormap
        norm = Normalize(vmin=frames.min(), vmax=frames.max())
        lc = LineCollection(segments, cmap=class_cmap, norm=norm)
        lc.set_array(frames)
        lc.set_linewidth(2)
        lc.set_label(f'{class_name} wheel position')

        ax8.add_collection(lc)
        ax8.autoscale()
        ax8.set_title('Wheel position', fontweight='bold')
        ax8.set_xlabel('$x$ position (pixels)')
        ax8.set_ylabel('$y$ position (pixels)')
        ax8.yaxis.set_inverted(True)
        # fig.colorbar(lc, ax=ax8, label='Frame Index')

        
    ax8.set_title('Wheel position', fontweight='bold')
    ax8.set_xlabel('$x$ position (pixels)')
    ax8.set_ylabel('$y$ position (pixels)')
    # ax8.legend()
    ax8.grid()


    fig.set_layout_engine('constrained')
    plt.show()





def create_csv(tracking_data, filtered_data, output_file):
    '''Create CSV with Frame, Feature 1 x, Feature 2 y, Feature 1 x, Feature 2 y, x difference, y difference, euclidean distance'''
    
    if len(tracking_data) < 2:
        print("Need at least 2 ellipses to create 4-column output")
        return
    
    # Get all available frames from both ellipses
    all_frames = set()
    for class_id in [0, 1]:
        if class_id in tracking_data:
            all_frames.update(tracking_data[class_id]['frames'])
    
    all_frames = sorted(all_frames)
    
    # Create dictionaries for quick lookup by frame
    ellipse0_data = {}
    ellipse1_data = {}
    
    # Populate ellipse 0 data
    if 0 in tracking_data:
        for i, frame in enumerate(tracking_data[0]['frames']):
            ellipse0_data[frame] = {
                'x': tracking_data[0]['x_pos'][i],
                'y': tracking_data[0]['y_pos'][i]
            }
    
    # Populate ellipse 1 data
    if 1 in tracking_data:
        for i, frame in enumerate(tracking_data[1]['frames']):
            ellipse1_data[frame] = {
                'x': tracking_data[1]['x_pos'][i],
                'y': tracking_data[1]['y_pos'][i]
            }

    

    # for j in ellipse0_data:
    #     if j in ellipse1_data:
    #         x_d = ellipse0_data[j]['x'] - ellipse1_data[j]['x']
    #         y_d = ellipse1_data[j]['y'] - ellipse0_data[j]['y'] 
    #         e_d = math.sqrt(x_d**2 + y_d**2)
    #     else:
    #         x_d = np.nan
    #         y_d = np.nan
    #         e_d = np.nan

    #     x_diff.append(x_d)
    #     y_diff.append(y_d)
    #     euclidean_diff.append(e_d)
    
    # Create the data structure

    # x_diff = []
    # y_diff = []
    # euclidean_diff = []

    data = []
    for frame in all_frames:
        # Get ellipse 0 data or use NaN if missing
        if frame in ellipse0_data:
            ellipse0_x = ellipse0_data[frame]['x']
            ellipse0_y = ellipse0_data[frame]['y']
        else:
            ellipse0_x = np.nan
            ellipse0_y = np.nan
        
        # Get ellipse 1 data or use NaN if missing
        if frame in ellipse1_data:
            ellipse1_x = ellipse1_data[frame]['x']
            ellipse1_y = ellipse1_data[frame]['y']
        else:
            ellipse1_x = np.nan
            ellipse1_y = np.nan

        if frame in ellipse0_data and frame in ellipse1_data:
            x_diff = ellipse0_data[frame]['x'] - ellipse1_data[frame]['x']
            y_diff = ellipse1_data[frame]['y'] - ellipse0_data[frame]['y'] 
            euclidean_diff = math.sqrt(x_diff**2 + y_diff**2)
        else:
            x_diff = np.nan
            y_diff = np.nan
            euclidean_diff = np.nan

        # x_diff.append(x_d)
        # y_diff.append(y_d)
        # euclidean_diff.append(e_d)
        
        row = [
            frame,
            ellipse0_x,
            ellipse0_y,
            ellipse1_x,
            ellipse1_y,
            x_diff,
            y_diff,
            euclidean_diff
        ]
        data.append(row)
    
    # Create DataFrame and save as CSV
    df = pd.DataFrame(data, columns=[
        'Frame', 
        'F_wheel_x', 
        'F_wheel_y', 
        'R_wheel_x', 
        'R_wheel_y',
        'X_Difference',
        'Y_Difference',
        'Euclidean_Distance'
    ])
    
    df.to_csv(output_file, index=False)
    print(f"CSV saved to: {output_file}")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {', '.join(df.columns)}")
    print(f"Frames with ellipse 0 data: {len(ellipse0_data)}")
    print(f"Frames with ellipse 1 data: {len(ellipse1_data)}")
    print(f"Total unique frames: {len(all_frames)}")
    print("\nFirst 10 rows:")
    print(df.head(10))
    
    # Show data completeness statistics
    print(f"\nData Completeness:")
    print(f"Ellipse 0: {df['F_wheel_x'].notna().sum()}/{len(df)} frames ({df['F_wheel_x'].notna().mean()*100:.1f}%)")
    print(f"Ellipse 1: {df['R_wheel_x'].notna().sum()}/{len(df)} frames ({df['R_wheel_x'].notna().mean()*100:.1f}%)")
    
    return df



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot ellipse data with customizable options.')
    parser.add_argument('directory', type=str, help='Directory containing YOLOv8 .txt files')
    parser.add_argument('--cutoff', type=float, default=2.0, help='Lowpass filter cutoff frequency')
    parser.add_argument('--fs', type=float, default=30.0, help='Sampling frequency')
    parser.add_argument('--raw', action='store_true', help='Show raw data along with filtered data')
    parser.add_argument('--absolute', action='store_true', help='Show absolute differences (default)')
    parser.add_argument('--signed', action='store_true', help='Show signed differences instead of absolute')
    parser.add_argument('--output', type=str, default='ellipse_features.csv', help='Output CSV filename')
    
    args = parser.parse_args()
    
    # Handle mutual exclusion for difference type
    if args.signed:
        absolute_diff = False
    else:
        absolute_diff = True  # Default to absolute differences
    
    tracking_data = process_directory(args.directory)
    filtered_data = apply_filters(tracking_data, args.cutoff, args.fs)
    create_csv(tracking_data, filtered_data, args.output)
    
    plot_data(tracking_data, filtered_data, show_raw=args.raw, absolute_diff=absolute_diff)


