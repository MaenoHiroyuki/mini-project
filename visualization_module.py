import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import re
import warnings

warnings.filterwarnings("ignore")

def convert_duration_to_minutes(duration_str):
    if isinstance(duration_str, str):
        hours = re.search(r'(\d+)h', duration_str)
        minutes = re.search(r'(\d+)min', duration_str)

        total_minutes = 0
        if hours:
            total_minutes += int(hours.group(1)) * 60
        if minutes:
            total_minutes += int(minutes.group(1))
        
        return total_minutes
    return None

def plot_release_year_distribution(data):
    plt.figure(figsize=(10, 6))
    sns.histplot(data['ReleaseYear'], kde=True, bins=20)
    plt.title('Distribution of Release Years')
    plt.xlabel('Release Year')
    plt.ylabel('Frequency')
    plt.show()

def plot_duration_distribution(data):
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Duration'], kde=True, bins=20)
    plt.title('Distribution of Movie Durations')
    plt.xlabel('Duration (Minutes)')
    plt.ylabel('Frequency')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def generate_all_distributions(data):
    print(data['Duration'].head())
    data['Duration'] = data['Duration'].apply(convert_duration_to_minutes)
    plot_release_year_distribution(data)
    plot_duration_distribution(data)

if __name__ == "__main__":
    data = pd.read_csv('D:\\SCSE2024\\mini-project\\data\\anova_movie_data_corrected.csv')
    print(data.columns)
    generate_all_distributions(data)

actor_data = [
    'Tom Hanks|Tim Allen|Don Rickles',
    'Robin Williams|Kirsten Dunst|Bonnie Hunt',
    'Walter Matthau|Jack Lemmon|Ann-Margret',
    'Whitney Houston|Angela Bassett|Loretta Devine',
    'Steve Martin|Diane Keaton|Martin Short',
    'Al Pacino|Robert De Niro|Val Kilmer',
    'Harrison Ford|Julia Ormond|Greg Kinnear',
    'Jonathan Taylor Thomas|Brad Renfro',
    'Jean-Claude Van Damme|Powers Boothe|Raymond J. Barry',
    'Pierce Brosnan|Sean Bean|Izabella Scorupco',
    'Michael Douglas|Annette Bening|Martin Sheen',
    'Leslie Nielsen|Mel Brooks|Peter MacNicol',
    'Kevin Bacon|Bob Hoskins|Bridget Fonda',
    'Anthony Hopkins|Joan Allen',
    'Geena Davis|Matthew Modine|Frank Langella',
    'Robert De Niro|Sharon Stone|Joe Pesci',
    'Emma Thompson|Kate Winslet|James Fleet',
    'Tim Roth|Antonio Banderas|Sammi Davis',
    'Jim Carrey',
    'Wesley Snipes|Woody Harrelson|Jennifer Lopez',
    'Gene Hackman|Rene Russo|Danny DeVito',
    'Sigourney Weaver|Holly Hunter|Dermot Mulroney',
    'Sylvester Stallone|Antonio Banderas|Julianne Moore',
    'Mary Steenburgen|Sean Patrick Flanery|Lance Henriksen'
]

data = pd.DataFrame({'Actors': actor_data})

data['NumActors'] = data['Actors'].apply(lambda x: len(x.split('|')))

print(data)

plt.figure(figsize=(10, 6))

sns.histplot(data['NumActors'], kde=True, bins=10)
plt.title('Distribution of Number of Actors per Movie')
plt.xlabel('Number of Actors')
plt.ylabel('Frequency')
plt.tight_layout()

plt.show()
