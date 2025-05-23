import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import matplotlib.cm as cm
import calendar

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

def plot_whole_timespan(dataframe, date_column, sales_column):
    df_copy = dataframe.copy()
    monthly_data = df_copy[sales_column].resample('M').sum()

    # Weißer Hintergrund für Figure und Axes
    fig, ax = plt.subplots(figsize=(16, 8), facecolor='white')
    ax.set_facecolor('white')

    ax.plot(monthly_data.index, monthly_data.values, 'b-', linewidth=2, label='Monatliche Verkäufe')

    # Richte die X-Achse explizit ein
    years = mdates.YearLocator(1)  # Jedes Jahr
    years_fmt = mdates.DateFormatter('%Y')
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)

    # X-Achsen-Label drehen für bessere Lesbarkeit
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Jahr', fontsize=12)
    ax.set_ylabel('Umsatz', fontsize=12)
    ax.set_title('Monatliche Verkaufsdaten (kompletter Zeitraum)', fontsize=15)

    # Durchschnittslinie
    avg = monthly_data.mean()
    ax.axhline(y=avg, color='r', linestyle='--', label=f'Durchschnitt: {avg:.2f}')

    ax.legend(loc='best')
    fig.tight_layout()
    plt.show()

    print(f"Zeitraum: {monthly_data.index.min().strftime('%Y-%m')} bis {monthly_data.index.max().strftime('%Y-%m')}")
    print(f"Anzahl der Monate: {len(monthly_data)}")
    print(f"Durchschnittlicher Umsatz: {avg:.2f}")
    print(f"Min: {monthly_data.min():.2f}, Max: {monthly_data.max():.2f}")

    return monthly_data


def plot_monthly_data_range(dataframe, sales_column, start_year=None, end_year=None):
    fig, ax = plt.subplots(figsize=(16, 8))
    title = 'Monatliche Verkaufsdaten'
    if start_year and end_year:
        title += f' ({start_year} bis {end_year})'
    ax.set_title(title, fontsize=15)
    df_copy = dataframe.copy()
    monthly_data = df_copy[sales_column].resample('M').sum()

    # Zeitraumfilterung, falls angegeben
    filtered_data = monthly_data
    if start_year and end_year:
        start_date = f'{start_year}-01-01'
        end_date = f'{end_year}-12-31'
        filtered_data = monthly_data[(monthly_data.index >= start_date) &
                                     (monthly_data.index <= end_date)]

    # Prüfen, ob Daten im angegebenen Zeitraum vorhanden sind
    if len(filtered_data) == 0:
        print(f"Warnung: Keine Daten für den Zeitraum {start_year or 'Anfang'} bis {end_year or 'Ende'} gefunden!")
        print(
            f"Verfügbarer Datenbereich: {monthly_data.index.min().strftime('%Y-%m')} bis {monthly_data.index.max().strftime('%Y-%m')}")

        # Zeige alle verfügbaren Daten stattdessen
        filtered_data = monthly_data
        ax.set_title(
            f'Monatliche Verkaufsdaten (verfügbar: {monthly_data.index.min().strftime("%Y-%m")} bis {monthly_data.index.max().strftime("%Y-%m")})',
            fontsize=15)

    # Plotting
    ax.plot(filtered_data.index, filtered_data.values, 'b-', linewidth=2, label='Monatliche Verkäufe')

    # Monatsgenau X-Achse einstellen
    months = mdates.MonthLocator()  # Jeder Monat
    months_fmt = mdates.DateFormatter('%Y-%m')  # Jahr-Monat Format

    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(months_fmt)

    # X-Achsen-Label rotieren für bessere Lesbarkeit
    plt.setp(ax.get_xticklabels(), rotation=90, ha='center')

    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Datum', fontsize=12)
    ax.set_ylabel('Umsatz', fontsize=12)

    # Durchschnittslinie
    avg = filtered_data.mean()
    ax.axhline(y=avg, color='r', linestyle='--', label=f'Durchschnitt: {avg:.2f}')

    ax.legend(loc='best')
    fig.tight_layout()
    plt.show()

    # Statistik
    print(
        f"Angezeigter Zeitraum: {filtered_data.index.min().strftime('%Y-%m')} bis {filtered_data.index.max().strftime('%Y-%m')}")
    print(f"Anzahl der Monate: {len(filtered_data)}")
    print(f"Durchschnittlicher Umsatz: {avg:.2f}")
    print(f"Min: {filtered_data.min():.2f}, Max: {filtered_data.max():.2f}")

    return filtered_data


def plot_yearly_sales_comparison(dataframe, date_column, sales_column, start_year=2005, end_year=2025):
    monthly_data = dataframe[sales_column].resample('M').sum()
    # Erstelle eine neue Figur mit ausreichender Größe
    plt.figure(figsize=(16, 10))

    # Liste für die Legende
    legend_entries = []

    # Farben für die Jahre generieren - Farbverlauf
    years_range = range(start_year, end_year + 1)
    num_years = len(years_range)

    # Erstelle eine benutzerdefinierte Farbpalette für bessere Unterscheidbarkeit
    # Farbverlauf von blau über grün, gelb, orange bis rot
    colors = cm.get_cmap('viridis', num_years)

    # Für jedes Jahr eine Kurve hinzufügen
    for i, year in enumerate(years_range):
        # Daten für das Jahr filtern
        year_data = monthly_data[monthly_data.index.year == year]

        if len(year_data) > 0:
            # Erstelle einen einheitlichen Datumsbereich für alle Jahre (Jan-Dez)
            # Verwende den Monat und Tag aus den gefilterten Daten, aber setze Jahr auf 2000
            norm_dates = [pd.Timestamp(2000, date.month, 1) for date in year_data.index]

            # Plotte die Kurve mit einer Farbe aus der Palette
            line, = plt.plot(norm_dates, year_data.values, linewidth=2,
                             alpha=0.8, color=colors(i / num_years))

            # Füge zur Legende hinzu
            legend_entries.append((line, f'{year} (max: {year_data.max():.0f})'))
        else:
            print(f"Keine Daten für das Jahr {year} gefunden.")

    # X-Achse formatieren auf Monatsbasis
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # Kurzform der Monate

    # Diagramm-Beschriftungen
    plt.title(f'Umsatzvergleich nach Jahren ({start_year}-{end_year})', fontsize=16)
    plt.xlabel('Monat', fontsize=14)
    plt.ylabel('Monatlicher Umsatz', fontsize=14)
    plt.grid(True, alpha=0.3)

    # Zwei-spaltige Legende außerhalb des Plots
    if legend_entries:
        plt.legend([entry[0] for entry in legend_entries],
                   [entry[1] for entry in legend_entries],
                   loc='upper center', bbox_to_anchor=(0.5, -0.15),
                   ncol=5, fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Platz für die Legende
    plt.show()

    # Statistik
    years_with_data = [year for year in years_range if len(monthly_data[monthly_data.index.year == year]) > 0]
    if years_with_data:
        print(f"Jahre mit verfügbaren Daten: {', '.join(map(str, years_with_data))}")

        # Durchschnittlicher, minimaler und maximaler Umsatz pro Jahr
        print("\nUmsatzstatistik pro Jahr:")
        for year in years_with_data:
            year_data = monthly_data[monthly_data.index.year == year]
            print(
                f"{year}: Durchschnitt: {year_data.mean():.2f}, Min: {year_data.min():.2f}, Max: {year_data.max():.2f}")
    else:
        print("Keine Daten im angegebenen Zeitraum gefunden.")

    return monthly_data


def plot_yearly_sales_comparison_with_zoom(dataframe, date_column, sales_column, start_year=2005, end_year=2025,
                                           zoom_months=(1, 4)):
    """
    Zeigt die Umsatzkurven für mehrere Jahre im Vergleich mit einem Zoom-Bereich für ausgewählte Monate.
    """
    # Kopie erstellen, um das Original nicht zu verändern
    df_copy = dataframe.copy()

    # Sicherstellen, dass das Datum als datetime vorliegt
    if date_column in df_copy.columns:
        if not pd.api.types.is_datetime64_any_dtype(df_copy[date_column]):
            try:
                df_copy[date_column] = pd.to_datetime(df_copy[date_column])
            except:
                print("Fehler: Konnte Datum nicht konvertieren")
                return None

        df_copy.set_index(date_column, inplace=True)

    # Monatliche Aggregation
    monthly_data = df_copy[sales_column].resample('M').sum()

    # Erstelle eine Figur mit zwei Subplot-Bereichen
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14), gridspec_kw={'height_ratios': [1, 1.5]})

    # Liste für die Legende
    legend_entries = []

    # Jahre-Bereich und Farbpalette
    years_range = range(start_year, end_year + 1)
    num_years = len(years_range)
    colors = cm.get_cmap('viridis', num_years)

    # Für jedes Jahr Daten plotten
    for i, year in enumerate(years_range):
        # Daten für das Jahr filtern
        year_data = monthly_data[monthly_data.index.year == year]

        if len(year_data) > 0:
            # Normalisierte Datumsbereich (alle Jahre auf 2000 gesetzt für Vergleichbarkeit)
            norm_dates = [pd.Timestamp(2000, date.month, date.day) for date in year_data.index]
            data_df = pd.DataFrame({'date': norm_dates, 'sales': year_data.values})

            # Farbe für das Jahr
            color = colors(i / num_years)

            # Plot im Hauptdiagramm - alle Monate
            line1, = ax1.plot(data_df['date'], data_df['sales'], linewidth=2,
                              alpha=0.8, color=color)

            # Für den Zoom-Bereich nur relevante Monate auswählen
            zoom_data = data_df[(data_df['date'].dt.month >= zoom_months[0]) &
                                (data_df['date'].dt.month <= zoom_months[1])]

            if not zoom_data.empty:
                # Plot im Zoom-Diagramm - nur ausgewählte Monate
                line2, = ax2.plot(zoom_data['date'], zoom_data['sales'], linewidth=2.5,
                                  alpha=0.9, color=color, marker='o', markersize=5)

            # Nur einen Legendeneintrag pro Jahr
            legend_entries.append((line1, f'{year} (max: {year_data.max():.0f})'))
        else:
            print(f"Keine Daten für das Jahr {year} gefunden.")

    # Formatierung für das Hauptdiagramm (ax1)
    ax1.set_title(f'Umsatzvergleich nach Jahren ({start_year}-{end_year})', fontsize=16)
    ax1.set_ylabel('Monatlicher Umsatz', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    # Formatierung für den Zoom-Bereich (ax2) - KORRIGIERTE BERECHNUNG
    zoom_start = pd.Timestamp(2000, zoom_months[0], 1)
    # Korrekte Berechnung des letzten Tags des Monats mit calendar
    _, last_day = calendar.monthrange(2000, zoom_months[1])
    zoom_end = pd.Timestamp(2000, zoom_months[1], last_day)

    # Explizit den Zoom-Bereich setzen
    ax2.set_xlim(zoom_start, zoom_end)

    # Diagramm-Titel und Labels
    month_names = [pd.Timestamp(2000, m, 1).strftime('%B') for m in range(zoom_months[0], zoom_months[1] + 1)]
    ax2.set_title(f'Zoom: {" bis ".join([month_names[0], month_names[-1]])}', fontsize=16)
    ax2.set_xlabel('Monat', fontsize=14)
    ax2.set_ylabel('Monatlicher Umsatz', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # Feinere X-Achsen-Ticks für den Zoom-Bereich
    ax2.xaxis.set_major_locator(mdates.MonthLocator())  # Ein Tick pro Monat
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%B'))  # Voller Monatsname
    ax2.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=0))  # Montage als kleinere Ticks

    # Zwei-spaltige Legende außerhalb der Plots
    if legend_entries:
        fig.legend([entry[0] for entry in legend_entries],
                   [entry[1] for entry in legend_entries],
                   loc='upper center', bbox_to_anchor=(0.5, 0),
                   ncol=5, fontsize=10)

    # Layout anpassen
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.15, hspace=0.3)

    # Zeige die Grafik
    plt.show()

    # Statistik für den Zoom-Bereich ausgeben
    print(f"\nStatistik für Zoom-Bereich (Monate {zoom_months[0]}-{zoom_months[1]}):")
    years_with_data = [year for year in years_range if len(monthly_data[monthly_data.index.year == year]) > 0]

    for year in years_with_data:
        year_data = monthly_data[monthly_data.index.year == year]
        zoom_data = year_data[(year_data.index.month >= zoom_months[0]) &
                              (year_data.index.month <= zoom_months[1])]

        if len(zoom_data) > 0:
            print(
                f"{year}: Durchschnitt: {zoom_data.mean():.2f}, Min: {zoom_data.min():.2f}, Max: {zoom_data.max():.2f}")

    return monthly_data


def plot_monthly_totals_across_years(dataframe, sales_column, start_year=2005, end_year=2025):
    df_copy = dataframe.copy()
    monthly_data = df_copy[sales_column].resample('M').sum()
    filtered_data = monthly_data[
        (monthly_data.index.year >= start_year) &
        (monthly_data.index.year <= end_year)
        ]

    if len(filtered_data) == 0:
        print(f"Keine Daten für den Zeitraum {start_year}-{end_year} gefunden!")
        return None

    # Extrahiere Monat aus dem Datum und gruppiere nach Monat
    monthly_totals = filtered_data.groupby(filtered_data.index.month).sum()

    # Sortiere nach Monat (1-12)
    monthly_totals = monthly_totals.reindex(range(1, 13))

    # Erstelle ein Balkendiagramm
    plt.figure(figsize=(14, 8))

    # Monatsnamen für die X-Achse
    month_names = [calendar.month_name[i] for i in range(1, 13)]

    # Farben mit einem farbenfrohen Farbverlauf
    colors = plt.cm.viridis(np.linspace(0, 0.8, 12))

    # Balkendiagramm mit Monatsdaten
    bars = plt.bar(month_names, monthly_totals, color=colors, width=0.7)

    # Werte über den Balken anzeigen
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                 f'{int(height):,}',
                 ha='center', va='bottom', fontsize=10)

    # Diagramm-Beschriftungen
    plt.title(f'Gesamtumsatz pro Monat über alle Jahre ({start_year}-{end_year})', fontsize=16)
    plt.xlabel('Monat', fontsize=14)
    plt.ylabel('Kumulierter Umsatz', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)

    # Layout anpassen
    plt.tight_layout()

    # Statistik anzeigen
    print(f"Gesamtumsatz pro Monat (Jahre {start_year}-{end_year}):")
    for i, total in enumerate(monthly_totals, 1):
        print(f"{calendar.month_name[i]}: {total:,.2f}")

    print(f"\nHöchster Monatsumsatz: {calendar.month_name[monthly_totals.idxmax()]} ({monthly_totals.max():,.2f})")
    print(f"Niedrigster Monatsumsatz: {calendar.month_name[monthly_totals.idxmin()]} ({monthly_totals.min():,.2f})")
    print(f"Durchschnittlicher Monatsumsatz: {monthly_totals.mean():,.2f}")

    # Zeige die Grafik
    plt.show()

    return monthly_totals

import matplotlib.pyplot as plt
import numpy as np
import calendar
import pandas as pd

def plot_monthly_average_across_years(dataframe, sales_column, start_year=2005, end_year=2025):
    df_copy = dataframe.copy()

    # 1. Resample auf Monatsbasis
    monthly_data = df_copy[sales_column].resample('M').sum()

    # 2. Filter nach Zeitraum
    filtered_data = monthly_data[
        (monthly_data.index.year >= start_year) &
        (monthly_data.index.year <= end_year)
    ]

    if len(filtered_data) == 0:
        print(f"Keine Daten für den Zeitraum {start_year}-{end_year} gefunden!")
        return None

    # 3. Durchschnitt pro Kalendermonat berechnen (z.B. alle Januare, alle Februare usw.)
    monthly_avg = filtered_data.groupby(filtered_data.index.month).mean()

    # 4. Reihenfolge der Monate sicherstellen
    monthly_avg = monthly_avg.reindex(range(1, 13))

    # 5. Plot erstellen
    plt.figure(figsize=(14, 8))

    # Monatsnamen
    month_names = [calendar.month_name[i] for i in range(1, 13)]

    # Farben z.B. mit Viridis-Farbskala
    colors = plt.cm.viridis(np.linspace(0, 0.8, 12))

    # Balkendiagramm
    bars = plt.bar(month_names, monthly_avg, color=colors, width=0.7)

    # Werte über den Balken anzeigen
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                 f'{height:,.2f}',
                 ha='center', va='bottom', fontsize=10)

    # Beschriftungen
    plt.title(f'Durchschnittlicher Monatsumsatz ({start_year}-{end_year})', fontsize=16)
    plt.xlabel('Monat', fontsize=14)
    plt.ylabel('Durchschnittlicher Umsatz', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    # Statistik ausgeben
    print(f"Durchschnittlicher Monatsumsatz ({start_year}-{end_year}):")
    for i, avg in enumerate(monthly_avg, 1):
        print(f"{calendar.month_name[i]}: {avg:,.2f}")

    print(f"\nHöchster Monatsdurchschnitt: {calendar.month_name[monthly_avg.idxmax()]} ({monthly_avg.max():,.2f})")
    print(f"Niedrigster Monatsdurchschnitt: {calendar.month_name[monthly_avg.idxmin()]} ({monthly_avg.min():,.2f})")
    print(f"Gesamtdurchschnitt über alle Monate: {monthly_avg.mean():,.2f}")

    # Anzeige
    plt.show()

    return monthly_avg

