import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Function to calculate area, perimeter, circularity, deformation, solidity, and mean intensity within a contour
def calculate_features(image, contour):
    
    # Calculate area and perimeter
    area = cv2.contourArea(contour)  #in micron
    perimeter = cv2.arcLength(contour, True) #in micron

    # Calculate circularity
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

    # Calculate deformation as 1 - circularity
    deformation = 1 - circularity

    # Create a mask for the contour
    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, color=255, thickness=cv2.FILLED)

    # Calculate mean intensity within the contour
    mean_intensity = cv2.mean(image, mask=mask)[0]

    # Calculate solidity (area / convex hull area)
    hull = cv2.convexHull(contour)  #in micron
    hull_area = cv2.contourArea(hull) 
    solidity = area / hull_area if hull_area > 0 else 0

    return area, perimeter, circularity, deformation, solidity, mean_intensity





# Function to get contours from text files
def get_contours_from_file(file_path):
    contours = []
    with open(file_path, 'r') as f:
        content = f.read()
        strings = content.split()
    for s in strings:
        points = [int(x) for x in s.split(',')]
        contour = np.array([(points[i], points[i + 1]) for i in range(0, len(points), 2)], dtype=np.int32)
        contour = contour.reshape((-1, 1, 2))
        contours.append(contour)
    return contours





# Function to check if a contour touches the edge of the image
def is_contour_touching_edge(contour, image_shape):
    # Get the image dimensions
    img_height, img_width = image_shape
    # Check each point in the contour
    for point in contour[:, 0, :]:  # contour is an array of shape (n, 1, 2)
        if point[0] == 0 or point[0] == img_width - 1 or point[1] == 0 or point[1] == img_height - 1:
            return True
    return False



# Function to plot the distribution of a variable
def plot_distribution(data, variable, subplot_index, num_rows, num_cols):
    plt.subplot(num_rows, num_cols, subplot_index)
    for dataset in data['Dataset'].unique():
        sns.histplot(data[data['Dataset'] == dataset][variable], kde=True, label=dataset, stat='density')
    plt.title(f'Distribution of {variable}')
    plt.xlabel(variable)
    plt.ylabel('Density')
    plt.legend()


# Function to plot the boxplot of a variable
def plot_boxplot_with_scatter(data, variable, subplot_index, num_rows):
    plt.subplot(num_rows, 1, subplot_index)
    sns.boxplot(x='Dataset', y=variable, data=data, palette="Set3")
    sns.stripplot(x='Dataset', y=variable, data=data, color='black', alpha=0.5)
    plt.title(f'{variable} - Mean and Standard Deviation')
    plt.ylabel(variable)
    plt.xlabel('Dataset')
    
    

# Function to remove outliers based on IQR for selected columns
def remove_selected_outliers(df, columns):
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df



# Function to plot the distribution of a variable for two groups side by side
def plot_distribution_side_by_side(data_a, data_b, variable, title_a, title_b, num_rows, num_cols, subplot_index):
    # Plot for Cell A
    plt.subplot(num_rows, num_cols, subplot_index)
    for dataset in data_a['Dataset'].unique():
        sns.histplot(data_a[data_a['Dataset'] == dataset][variable], kde=True, label=dataset, stat='density')
    plt.title(title_a)
    plt.xlabel(f'{variable}')
    plt.ylabel('Density')
    plt.legend()

    # Plot for Cell B
    plt.subplot(num_rows, num_cols, subplot_index + 1)
    for dataset in data_b['Dataset'].unique():
        sns.histplot(data_b[data_b['Dataset'] == dataset][variable], kde=True, label=dataset, stat='density')
    plt.title(title_b)
    plt.xlabel(f'{variable}')
    plt.ylabel('Density')
    plt.legend()
    
    
# Function to plot boxplots of a variable for two groups side by side  
def plot_boxplots_side_by_side(data_a, data_b, variable, title_a, title_b, num_rows, num_cols, subplot_index):
    # Box plot for Cell A
    plt.subplot(num_rows, num_cols, subplot_index)
    sns.boxplot(x='Dataset', y=variable, data=data_a, palette="Set3")
    sns.stripplot(x='Dataset', y=variable, data=data_a, color='black', alpha=0.5)
    plt.title(title_a)
    plt.xlabel('Dataset')
    plt.ylabel(f'{variable}')

    # Box plot for Cell B
    plt.subplot(num_rows, num_cols, subplot_index + 1)
    sns.boxplot(x='Dataset', y=variable, data=data_b, palette="Set3")
    sns.stripplot(x='Dataset', y=variable, data=data_b, color='black', alpha=0.5)
    plt.title(title_b)
    plt.xlabel('Dataset')
    plt.ylabel(f'{variable}')



    
# Function to calculate mean, median, and standard deviation
def calculate_stats(data, feature):
    mean_val = data[feature].mean()
    median_val = data[feature].median()
    std_dev = data[feature].std()
    return mean_val, median_val, std_dev



# Function to calculate percentage change
def calculate_percentage_change(old_value, new_value):
    if old_value != 0:
        return (new_value - old_value) / old_value * 100
    else:
        return None  # Avoid division by zero

    

# Function to print descriptive sentences for percentage changes
def print_change_description(cell_type, change, metric):
    change_description = "increased" if change > 0 else "decreased"
    print(f"The {metric} values in {cell_type} datasets have {change_description} by {abs(change):.2f}%.")
    
    
# Function to check p value    
def interpret_p_value(p_value, alpha=0.05):
    if p_value < alpha:
        return "statistically significant."
    else:
        return "not statistically significant."
    
# Function to print the frames correspondent to outlier in the data
def plot_extreme_values_frames(dataframe, feature, dataset_name, segmentation_dir, num_images=25):
    # Sorting by feature to get lowest and highest values
    sorted_df = dataframe.sort_values(by=[feature])
    lowest_values = sorted_df.head(num_images // 2)
    highest_values = sorted_df.tail(num_images - num_images // 2)
    
    # Combine lowest and highest into a single DataFrame
    combined_df = pd.concat([lowest_values, highest_values])
    
    # Create subplots
    num_rows = 5
    num_cols = 5
    plt.figure(figsize=(15, 3 * num_rows))
    
    for i, (index, row) in enumerate(combined_df.iterrows()):
        frame_number = int(row['Frame'])
        image_path = os.path.join(segmentation_dir, dataset_name, 'outlines', f'frame_{frame_number:04d}_outlines.png')
        image = cv2.imread(image_path)
        
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f'Frame {frame_number}')
        plt.axis('off')
    
    plt.suptitle(f'Extremes for {feature} - {dataset_name}', fontsize=16)
    plt.tight_layout()
    plt.show()
    
def plot_scatter_matrix(data, metrics):
    plt.figure(figsize=(15, 15))
    plot_count = 1
    for i in range(len(metrics)):
        for j in range(i+1, len(metrics)):
            metric1, metric2 = metrics[i], metrics[j]

            plt.subplot(len(metrics)-1, len(metrics)-1, plot_count)
            plot_count += 1

            for label, df in data.items():
                sns.scatterplot(x=df[metric1], y=df[metric2], label=label)

            plt.xlabel(metric1)
            plt.ylabel(metric2)
            plt.title(f"{metric1} vs {metric2}")

    plt.tight_layout()
    plt.show()
    
    
def plot_scatter_matrix_mean(data, metrics):
    plt.figure(figsize=(15, 15))
    plot_count = 1
    for i in range(len(metrics)):
        for j in range(i+1, len(metrics)):
            metric1, metric2 = metrics[i], metrics[j]

            plt.subplot(len(metrics)-1, len(metrics)-1, plot_count)
            plot_count += 1

            for label, df in data.items():
                # Calculate the mean (or central point) for each metric
                mean_x = df[metric1].mean()
                mean_y = df[metric2].mean()

                # Plot the mean point for each dataset
                plt.scatter(mean_x, mean_y, label=f"{label} Mean", s=100, edgecolor='black')

            plt.xlabel(metric1)
            plt.ylabel(metric2)
            plt.title(f"{metric1} vs {metric2}")
            plt.legend()

    plt.tight_layout()
    plt.show()
    
    
    
    
  