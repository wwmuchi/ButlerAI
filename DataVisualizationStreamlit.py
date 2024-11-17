import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


st.subheader("Data Review")

### Import data
df_mast = pd.read_csv('s2_aam_raw_data.csv', index_col = 0) # master data set
df_imp = df_mast # imported data set

# Add and delete columns to test the code below
df_imp['this_is_a_test_col'] =  np.random.rand(len(df_imp)) # add


st.write("Here is your imported data (first 50 rows):")
st.write(df_imp.head(n=50))

### STEP 1: Identify missing and additional columns

# Define ideal columns
ideal_columns = ['employeeid', 'gender', 'race', 'age', 'experience', 
                 'education', 'job_level', 'department', 'entry_date', 
                 'entry_wage', 'year', 'perf_review', 'job_change_reason', 
                 'salary', 'leave_date', 'managerid', 'manager_race', 
                 'manager_gender', 'applicantid', 'job_source', 
                 'location', 'job_title']

# Pull cols in imported data
df0_columns = list(df_imp.columns)

# Determine missing and additional columns
cols_missing = [col for col in ideal_columns if col not in df0_columns]
cols_additional = [col for col in df0_columns if col not in ideal_columns]
cols_no_info = [col for col in df_imp.columns if df_imp[col].isnull().all()]
cols_available = [col for col in ideal_columns if col in df0_columns]

# Display missing and additional columns as text, if needed
if cols_missing:
    missing_columns = "\n".join([f"- {col}" for col in cols_missing])
    st.write("The following variables are missing in your data:")
    st.markdown(missing_columns)

# Display unnecessary columns and ask if the user wants to drop them
if cols_additional:
    additional_columns = "\n".join([f"- {col}" for col in cols_additional])
    st.write("The following columns are not in the master data set:")
    st.markdown(additional_columns)

if cols_no_info:
    no_info_columns = "\n".join([f"- {col}" for col in cols_no_info])
    st.write("The following columns have no information:")
    st.markdown(no_info_columns)


st.write("Would you like to drop these columns?")
    
# Multi-select for choosing specific columns to drop
columns_to_drop = st.multiselect("Select columns to drop:", options=set(cols_additional + cols_no_info))

# Drop selected columns if any were chosen
if columns_to_drop:
    # Drop only the selected columns
    df_imp = df_imp.drop(columns=columns_to_drop)
    st.write("Selected columns have been dropped.")
else:
    st.write("No columns were dropped.")


### STEP 3: Ask user if would like to drop rows with no applicant or employee id

# Identify rows with no applicant or employee id
rows_no_id = df_imp[(df_imp['applicantid'].isnull()) & (df_imp['employeeid'].isnull())]

# Ask to drop rows with no id
if not rows_no_id.empty:
    st.write("The following rows have no applicant or employee id:")
    st.write(rows_no_id)
    drop_rows_no_id = st.radio("Drop rows with no id?", ("Yes", "No"))
    
    if drop_rows_no_id == "Yes":
        df_imp = df_imp.drop(index=rows_no_id.index)
        st.write("Rows with no id have been dropped.")
    else:
        st.write("No rows were dropped.")



### Step 4: Identify variables to be visualized

st.subheader("Visualization")
visualize_data = st.radio('Would you like to visualize the data?', ("Yes", "No"), index=1)

if visualize_data == "Yes":
    pass
if visualize_data == "No":
    st.stop()



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
    'Performance Review': 'perf_review',
    'Job Change Reason': 'job_change_reason'
}

# Determine which variables are available in the data
var_dict_able_to_visualize = {title: var for title, var in var_dict_able_to_visualize.items() if var in df_imp.columns}

# Ask the user which plots they would like to visualize
selected_variables = st.multiselect("Choose variables you would like to visualize:", options=var_dict_able_to_visualize.keys())
var_dict_selected = {title: var for title, var in var_dict_able_to_visualize.items() if title in selected_variables}



### Step 6: Group variables into graph types

# Vars in master data set
category_vars = ['gender', 'race', 'education', 'department',
                 'experience', 'job_level', 'manager_race', 'manager_gender',
                'location', 'job_title']
binned_category_vars = ['salary', 'entry_wage', 'perf_review', 'age']
vars_to_count = ['job_change_reason']
time_vars = ['entry_date', 'leave_date']


# Vars selected in imported data set
category_vars_selected = {title: var for title, var in var_dict_selected.items() if var in category_vars}
binned_category_vars_selected = {title: var for title, var in var_dict_selected.items() if var in binned_category_vars}
vars_to_count_selected = {title: var for title, var in var_dict_selected.items() if var in vars_to_count}
time_vars_selected = {title: var for title, var in var_dict_selected.items() if var in time_vars}


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


    # Display distribution or count?
    distrib_or_count = st.radio("Would you like to visualize the distribution or raw count?", ("Distribution", "Count"))



else:
    dataframes = []



### Step 7: Create time variables

if time_vars_selected:
    # Ask user how to bin time variables
    time_unit = st.radio("How would you like to display time variables?", ("Year", "Month", "Day"))

    # Pull unit of choice
    for title, feature in time_vars_selected.items():

        # Add time of interest to
        for df, _ in dataframes:
            date_time = pd.to_datetime(df[feature].dropna(), format='%Y-%m-%d', errors='raise')
            df[f'{title} {time_unit}'] = date_time.dt.to_period(time_unit[0]).astype(str)
                
    # Add new col to dict with same title key as time_vars_selected
    time_vars_selected_post = {title: [f'{title} {time_unit}'] for title, feature in time_vars_selected.items()}

else:
    time_vars_selected_post = {}

### Step 8: Create binned variables

if binned_category_vars_selected:

    # Ask user how many bins
    names_to_bin = list((binned_category_vars_selected.keys()))

    num_bins = st.slider(
        f"Select the number of bins to group {', '.join(names_to_bin[:-1])} and {names_to_bin[-1]} into",
        2, 20, 10
    )
    
    # Apply binning to each variable
    for title, feature in binned_category_vars_selected.items():

        # Determine range for binning
        min_value = min(df_mast[feature].min(), df_imp[feature].min())
        max_value = max(df_mast[feature].max(), df_imp[feature].max())
        # Define bins and labels
        if feature == 'age':
            bins = np.linspace(min_value, max_value, num=num_bins).astype(int)
            labels = [f"{int(bins[i])} - {int(bins[i+1])}" for i in range(len(bins) - 1)]
        else:
            bins = np.linspace(min_value, max_value, num=num_bins)

        if max_value >= 1000:
            labels = [f"{bins[i] / 1000:.0f}K - {bins[i + 1] / 1000:.0f}K" for i in range(len(bins) - 1)]
        else:
            labels = [f"{bins[i]:.2f} - {bins[i + 1]:.2f}" for i in range(len(bins) - 1)]

        # Apply binning to each DataFrame
        for df, _ in dataframes:
            df[f'{feature} binned'] = pd.cut(df[feature], bins=bins, labels=labels, include_lowest=True)


    # Add binned feature to new dict with same key as binned_category_vars_selected
    binned_category_vars_selected_post = {title: [f'{feature} binned'] for title, feature in binned_category_vars_selected.items()}

else:
    binned_category_vars_selected_post = {}

### Step 9: Create vars to count


vars_to_count_selected_post = {}

# Calculate counts
for df, label in dataframes:

    for title, var in vars_to_count_selected.items():

        vars_to_count_selected_post[title] = []

        # Pull unique entries
        unique_entries = df[var].dropna().unique()

        # Add count of each unique entry to the DataFrame
        for entry in unique_entries:
            for id in id_list:
                entry_count = df[df[var] == entry].groupby(id).size().reset_index(drop=True)
                count_var_name = f'{var} {entry} count by {id}'
                df[count_var_name] = entry_count
                vars_to_count_selected_post[title].append(count_var_name)

# Step 10: convert item in category_vars_selected to list
category_vars_selected_post = {title: [var] for title, var in category_vars_selected.items()}


### Step: Group all selected variables into a single dictionary
selected_variables_post = {**category_vars_selected_post, **binned_category_vars_selected_post, **vars_to_count_selected_post, **time_vars_selected_post}


### Step 11: Determine which to plot as distribution or count



### Step 9: Graph

for title, cols in selected_variables_post.items():

    for col in cols:
        values = set().union(*(df[0][col].dropna() for df in dataframes))

        str_values = sorted(v for v in values if isinstance(v, str))
        float_values = sorted(v for v in values if isinstance(v, float))
        unique_values = str_values + float_values

        # Initialize plot_data
        plot_data_count = pd.DataFrame({col: unique_values})
        plot_data_proportion = pd.DataFrame({col: unique_values})


        # Remove duplicates and filter for applicants and employees for each dataframe
        for df, label in dataframes:
            df.drop_duplicates(subset= id_list + [col], keep='first', inplace=True)

            if identify_by_app_or_emp == "Applicant":
                df_applicants = df[df['applicantid'].notna()]
                count_by_applicants = df_applicants[col].value_counts().sort_index()
                plot_data_count[f'Applicants {label}'] = count_by_applicants.reindex(unique_values, fill_value=0).values

            if identify_by_app_or_emp == "Employee":
                df_employees = df[df['employeeid'].notna()]
                count_by_employees = df_employees[col].value_counts().sort_index()
                plot_data_count[f'Employees {label}'] = count_by_employees.reindex(unique_values, fill_value=0).values

            if identify_by_app_or_emp == "Both":
                df_applicants = df[df['applicantid'].notna()]
                count_by_applicants = df_applicants[col].value_counts().sort_index()
                plot_data_count[f'Applicants {label}'] = count_by_applicants.reindex(unique_values, fill_value=0).values

                df_employees = df[df['employeeid'].notna()]
                count_by_employees = df_employees[col].value_counts().sort_index()
                plot_data_count[f'Employees {label}'] = count_by_employees.reindex(unique_values, fill_value=0).values


        plot_data_proportion = plot_data_count.apply(lambda x: x / x.sum() if pd.api.types.is_integer_dtype(x) else x, axis=0)



    ############# PLOT V2 #############

    if distrib_or_count == "Distribution":
        graph_type = [(plot_data_proportion, 'Proportion')]
    elif distrib_or_count == "Count":
        graph_type = [(plot_data_count, 'Count')]


    for plot_data, count_v_distrib in graph_type:

        # Set up positions for bars with a smaller width to fit all four
        x = np.arange(len(unique_values))
        width = 0.2  # Narrower width for each bar to avoid overlap

        fig, ax = plt.subplots(figsize=(12, 6))

        def plot_applicants(ax):
            colors_applicants = ['blue', 'green']
            for i, (df, label) in enumerate(dataframes):
                ax.bar(x - (1.5 - i) * width, plot_data[f'Applicants {label}'], width,
                        label=f'Applicants ({label})', color=colors_applicants[i])
            ax.set_ylabel(f'Applicants {count_v_distrib}', color='blue')
            ax.tick_params(axis='y', labelcolor='blue')


        def plot_employees(ax):
            colors_employees = ['orange', 'red']
            for i, (_, label) in enumerate(dataframes):
                ax.bar(x + (0.5 + i) * width, plot_data[f'Employees {label}'], width,
                        label=f'Employees ({label})', color=colors_employees[i], alpha=0.7)
            ax.set_ylabel(f'Employees {count_v_distrib}', color='red')
            ax.tick_params(axis='y', labelcolor='red')


        ax.set_xlabel(col.title())

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

        if identify_by_app_or_emp == "Employee":
            plot_employees(ax)
            graph_intro_title = 'Applicants Grouped by'
            handles, labels = ax.get_legend_handles_labels()

        if identify_by_app_or_emp == "Both":
            ax1 = ax
            ax2 = ax1.twinx()
            plot_applicants(ax1)
            plot_employees(ax2)
            graph_intro_title = 'Applicants and Employees Grouped by'
            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            handles, labels = handles1 + handles2, labels1 + labels2


        plt.title(f"{graph_intro_title} {title.title()}", fontsize=16)

        # Combine legends
        fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0, 1), bbox_transform=ax.transAxes)

        # Rotate x-ticks for clarity if needed
        if len(unique_values) > 10:
            # for ax1 in [ax1, ax2]:
            ax1.tick_params(axis='x', rotation=90)

        st.pyplot(fig)