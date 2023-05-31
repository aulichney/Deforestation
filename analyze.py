def plot_feature_importances(FOLDER_NAME, use_abs = True):
    base_path  = 'FeatureImportanceResults_Lasso/'
    extension = 'lasso'

    file_path = base_path + FOLDER_NAME + '/' + extension + '.csv'

    df = pd.read_csv(file_path, index_col=0)

    # Calculate the sum of absolute values
    abs_sum = df['Coeff'].abs().sum()
    # Normalize the values in the column
    df['Coeff'] = df['Coeff'] / abs_sum

    # Select the first 10 values from the 'Coeff' and 'Feature' columns
    coeff_values = df['Coeff'].head(10)
    feature_labels = df['Feature'].head(10)

    if abs:
        coeff_values = abs(coeff_values)

    sns.set_style('whitegrid')
    plt.figure(figsize=(10, 6))
    sns.barplot(x=coeff_values, y=feature_labels, color='green')

    # Set plot title and labels
    plt.title(FOLDER_NAME + ' ' + extension.upper() )
    plt.xlabel('Abs')
    #plt.ylabel('Feature')
    plt.show()

FOLDER_NAME = '2004_2005_2006_PREDICT_2007'
plot_feature_importances(FOLDER_NAME, use_abs = True)

file_path = 'performance.txt'

lines=[]

with open(file_path, 'r') as file:
    for line in file:
        lines.append(line.strip())

df = pd.DataFrame([l.split(':') for l in lines[2:]], columns = ['Method', 'MSE'])
df.Method = df.Method.apply(lambda x: x.split(' ')[0])

df.MSE = df.MSE.apply(lambda x: float(x.strip()))

mse_values = df['MSE']
method_labels = df['Method']

sns.set_style('whitegrid')
plt.figure(figsize=(10, 6))

sns.barplot(x=mse_values, y=method_labels, color='green')

# Set plot title and labels
plt.title('MSE by Method')
#plt.xlabel('Abs')
#plt.ylabel('Feature')
plt.show()
