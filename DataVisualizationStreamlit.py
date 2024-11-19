import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.subheader("Visualization")

### Import data
df_mast = pd.read_csv('s2_aam_raw_data.csv', index_col = 0) # master data set
df_imp = df_mast.copy() # imported data set

### Step 4: Identify variables to be visualized


# Define variables that can be visualized
var_dict_able_to_visualize = {
    'Initial Entry Date': 'entry_date',
    'Final Exit Date': 'leave_date',
    'Race': 'race',
    'Gender': 'gender',
    'Age': 'age',
    'Education': 'education',
    'Department': 'department',
    'Salary': 'salary',
    'Entry Wage': 'entry_wage',
    'Performance Review': 'perf_review',
    'Job Change Reason': 'job_change_reason'
}

# Determine which variables are available in the data
var_dict_able_to_visualize = {title: var for title, var in var_dict_able_to_visualize.items() if var in df_imp.columns}

# Ask the user which plots they would like to visualize
selected_variables = st.multiselect("Choose variables you would like to visualize:", options=var_dict_able_to_visualize.keys())
var_dict_selected = {title: var for title, var in var_dict_able_to_visualize.items() if title in selected_variables}

### Step 5: Ask user how to display data
if var_dict_selected:

    # By which ID?
    identify_by_app_or_emp = st.radio("Would you like to identify by Applicant ID or Employee ID?", ("Applicant", "Employee", "Both"))
    
    if identify_by_app_or_emp == "Applicant":
        id_list = ['applicantid']
    elif identify_by_app_or_emp == "Employee":
        id_list = ['employeeid']
    else:
        id_list = ['applicantid', 'employeeid']
        
    # Compare to master data set?
    compare_to_master = st.radio("Would you like to compare to master data set?", ("Yes", "No"))

    # Define dataframes and labels for each dataset
    dataframes_both = [(df_mast, 'Master Data Set'), (df_imp, 'Imported Data Set')]
    dataframes_imported = [(df_imp, 'Imported Data Set')]

    # Set the dataframes based on the user's choice
    if compare_to_master == "Yes":
        dataframes = dataframes_both
    else:
        dataframes = dataframes_imported

    # Choose colors?
    colors = st.radio("Would you like to choose the colors for the graph?", ("Yes", "No"), index=1)


else:
    dataframes = []
    colors = "No"


### Step 6: Group variables into graph types

binned_category_vars = ['salary', 'entry_wage', 'perf_review', 'age']
time_vars = ['entry_date', 'leave_date']



##### Step: Ask user which colors they would like to use for the graph

# Define functions to mix colors
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')  # Remove the '#' character
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb_color):
    return '#{:02x}{:02x}{:02x}'.format(*rgb_color)

def mix_colors(color1, color2):
    rgb1 = hex_to_rgb(color1)
    rgb2 = hex_to_rgb(color2)
    mixed_rgb = tuple((rgb1[i] + rgb2[i]) // 2 for i in range(3))
    return rgb_to_hex(mixed_rgb)


# Set detault colors:
applicant_color_master_default = "#1f77b4"
employee_color_master_default = "#ff7f0e"
applicant_color_imported_default = "#2ca02c"
employee_color_imported_default = "#d62728"

if colors == "Yes":

    if compare_to_master == "Yes":

        if identify_by_app_or_emp == "Applicant":
            applicant_color_master = st.color_picker("Choose a color for Applicants in Master Data Set", applicant_color_master_default)
            applicant_color_imported = st.color_picker("Choose a color for Applicants in Imported Data Set", applicant_color_imported_default)
            applicant_axis_color = mix_colors(applicant_color_master, applicant_color_imported) 

        if identify_by_app_or_emp == "Employee":
            employee_color_master = st.color_picker("Choose a color for Employees in Master Data Set", employee_color_master_default)
            employee_color_imported = st.color_picker("Choose a color for Employees in Imported Data Set", employee_color_imported_default)
            employee_axis_color = mix_colors(employee_color_master, employee_color_imported)

        if identify_by_app_or_emp == "Both":
            applicant_color_master = st.color_picker("Choose a color for Applicants in Master Data Set", applicant_color_master_default)
            employee_color_master = st.color_picker("Choose a color for Employees in Master Data Set", employee_color_master_default)
            applicant_color_imported = st.color_picker("Choose a color for Applicants in Imported Data Set", applicant_color_imported_default)
            employee_color_imported = st.color_picker("Choose a color for Employees in Imported Data Set", employee_color_imported_default)
            applicant_axis_color = mix_colors(applicant_color_master, applicant_color_imported) 
            employee_axis_color = mix_colors(employee_color_master, employee_color_imported)


    if compare_to_master == "No":

        if identify_by_app_or_emp == "Applicant":
            applicant_color_imported = st.color_picker("Choose a color for Applicants in Imported Data Set", "#2ca02c")
        if identify_by_app_or_emp == "Employee":
            employee_color_imported = st.color_picker("Choose a color for Employees in Imported Data Set", "#d62728")
        if identify_by_app_or_emp == "Both":
            applicant_color_imported = st.color_picker("Choose a color for Applicants in Imported Data Set", "#2ca02c")
            employee_color_imported = st.color_picker("Choose a color for Employees in Imported Data Set", "#d62728")
            applicant_axis_color = applicant_color_imported
            employee_axis_color = employee_color_imported

else:
    applicant_color_imported = applicant_color_imported_default
    employee_color_imported = employee_color_imported_default
    applicant_color_master = applicant_color_master_default
    employee_color_master = employee_color_master_default
    applicant_axis_color = mix_colors(applicant_color_master, applicant_color_imported)
    employee_axis_color = mix_colors(employee_color_master, employee_color_imported)



### Step 7: Define function that transforms time variables


def time_vars_transform(time_series, existing_time_unit = None):

    if existing_time_unit:
        time_unit = existing_time_unit
    else:
        time_unit = st.radio(f"How would you like to display {title}?", ("Year", "Month", "Day"), key=f'{title} time unit')

    date_time = pd.to_datetime(time_series.dropna(), format='%Y-%m-%d', errors='raise')
    time_series_transformed = date_time.dt.to_period(time_unit[0]).astype(str)

    return time_series_transformed, time_unit

### Step 8: Define function that transforms binned variables

def binned_vars_transform(binned_series, existing_bins = None, existing_labels = None): # binned df needs to have a column for master and imported data so that same bins are applied to both
    
    col = binned_series.name

    if existing_bins is not None and existing_labels is not None:
        bins = existing_bins
        labels = existing_labels
    
    else:
        num_bins = st.slider(
        f"Select the number of bins to group {title} into",
        2, 20, 10,
        key=f'{col} num bins'
        )

        values = set().union(*(df[col].dropna() for df, _ in dataframes))

        min_value = min(values)
        max_value = max(values)

        # Define bins and labels
        if title == 'Age':
            bins = np.linspace(min_value, max_value, num=num_bins).astype(int)
            labels = [f"{int(bins[i])} - {int(bins[i+1])}" for i in range(len(bins) - 1)]
        else:
            bins = np.linspace(min_value, max_value, num=num_bins)

        if max_value >= 1000:
            labels = [f"{bins[i] / 1000:.0f}K - {bins[i + 1] / 1000:.0f}K" for i in range(len(bins) - 1)]
            labels = pd.Categorical(labels, ordered=True, categories=labels)

        else:
            labels = [f"{bins[i]:.2f} - {bins[i + 1]:.2f}" for i in range(len(bins) - 1)]    



    # Apply binning  
      
    series_to_return = pd.cut(binned_series, bins=bins, labels=labels, include_lowest=False)
    
    return series_to_return, bins, labels


### Step 9: Define a function that turns columns into counts by unique values

def histogram_transform(col):

    values = set().union(*(df[col].dropna() for df, _ in dataframes))

    str_values = sorted(v for v in values if isinstance(v, str))
    str_values_no_num = [v for v in str_values if not any(char.isdigit() for char in v)]
    str_values_num = [v for v in str_values if any(char.isdigit() for char in v)]
    str_values_num = sorted(str_values_num, key=lambda x: float(''.join(filter(str.isdigit, x))))
    float_values = sorted(v for v in values if isinstance(v, float))
    unique_values = str_values_no_num + str_values_num  + float_values

    # Initialize plot_data
    plot_data = pd.DataFrame({col: unique_values})

    # Remove duplicates and filter for applicants and employees for each dataframe
    for df, label in dataframes:
        df.drop_duplicates(subset= id_list + [col], keep='first', inplace=True)

        if identify_by_app_or_emp == "Applicant":
            df_applicants = df[df['applicantid'].notna()]
            count_by_applicants = df_applicants[col].value_counts().sort_index()
            plot_data[f'Applicants {label}'] = count_by_applicants.reindex(unique_values, fill_value=0).values

        if identify_by_app_or_emp == "Employee":
            df_employees = df[df['employeeid'].notna()]
            count_by_employees = df_employees[col].value_counts().sort_index()
            plot_data[f'Employees {label}'] = count_by_employees.reindex(unique_values, fill_value=0).values

        if identify_by_app_or_emp == "Both":
            df_applicants = df[df['applicantid'].notna()]
            count_by_applicants = df_applicants[col].value_counts().sort_index()
            plot_data[f'Applicants {label}'] = count_by_applicants.reindex(unique_values, fill_value=0).values

            df_employees = df[df['employeeid'].notna()]
            count_by_employees = df_employees[col].value_counts().sort_index()
            plot_data[f'Employees {label}'] = count_by_employees.reindex(unique_values, fill_value=0).values

    return plot_data, unique_values


### Step: Define plot functions

def width_calculator(i, sign):
    if identify_by_app_or_emp == "Both":
        bar_widths = x + (sign)*(i+1/2)*width
    else:
        if compare_to_master == "Yes":
            j = -1 if i == 0 else 1
            bar_widths = x + width*(j/2)
        else:
            bar_widths = x
    return bar_widths


# Define plot functions
def plot_applicants(ax):
    for i, (_, label) in enumerate(dataframes):
        bar_widths = width_calculator(i, 1)
        ax.bar(bar_widths, plot_data[f'Applicants {label}'], width,
                label=f'Applicants ({label})', color= applicant_color_master if label == 'Master Data Set' else applicant_color_imported)
    ax.set_ylabel(f'Applicants Count', color=applicant_axis_color)
    ax.tick_params(axis='y', labelcolor=applicant_axis_color)

def plot_employees(ax):
    for i, (_, label) in enumerate(dataframes):
        bar_widths = width_calculator(i,-1)
        ax.bar(bar_widths, plot_data[f'Employees {label}'], width,
                label=f'Employees ({label})', color= employee_color_master if label == 'Master Data Set' else employee_color_imported)
    ax.set_ylabel(f'Employees Count', color=employee_axis_color)
    ax.tick_params(axis='y', labelcolor=employee_axis_color)


### Step: Plot


for title, col in var_dict_selected.items():
    

    if col in time_vars:
        df_imp[col], time_unit = time_vars_transform(df_imp[col])
        df_mast[col], _ = time_vars_transform(df_mast[col], time_unit)

    if col in binned_category_vars:
        df_imp[col], bins, labels = binned_vars_transform(df_imp[col])
        df_mast[col], _, _ = binned_vars_transform(df_mast[col], bins,labels)
    
    plot_data, unique_values = histogram_transform(col)

    # Set up positions for bars
    x = np.arange(len(unique_values))
    width = 0.13

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.set_xlabel(title)

    if len(unique_values) > 30:
        step = len(unique_values) // 30
        displayed_ticks = unique_values[::step]
        x_ticks = x[::step]
    else:
        displayed_ticks = unique_values
        x_ticks = x

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(displayed_ticks)
    ax.grid(True, linestyle='--', alpha=0.5)

    # Plot based on `plot_type`
    if identify_by_app_or_emp == "Applicant":
        plot_applicants(ax)
        graph_intro_title = 'Applicants Grouped by'
        handles, labels = ax.get_legend_handles_labels()
        if len(unique_values) > 10:
            ax.tick_params(axis='x', rotation=90)

    if identify_by_app_or_emp == "Employee":
        plot_employees(ax)
        graph_intro_title = 'Applicants Grouped by'
        handles, labels = ax.get_legend_handles_labels()
        if len(unique_values) > 10:
            ax.tick_params(axis='x', rotation=90)

    if identify_by_app_or_emp == "Both":
        ax1 = ax
        ax2 = ax1.twinx()
        plot_applicants(ax1)
        plot_employees(ax2)
        graph_intro_title = 'Applicants and Employees Grouped by'
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles, labels = handles1 + handles2, labels1 + labels2
        if len(unique_values) > 10:
            ax1.tick_params(axis='x', rotation=90)

    plt.title(f"{graph_intro_title} {title.title()}", fontsize=16)

    # Combine legends
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0, 1), bbox_transform=ax.transAxes)

    # Rotate x-ticks for clarity if needed
    if len(unique_values) > 10:
        ax.tick_params(axis='x', rotation=90)

    st.pyplot(fig)