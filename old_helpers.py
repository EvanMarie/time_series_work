import pandas as pd
import numpy as np
import matplotlib as mpl
from IPython.core.display import HTML
import matplotlib.pyplot as plt

pd.options.display.float_format = '{:,.2f}'.format

pretty_label_text = 'black'
pretty_label_background = '#E1C7C2'
pretty_background = '#DBE5EB'
pretty_text = 'black'
bgcolor = "#E1C7C2";
text_color = 'black'
innerbackcolor = "#1E3A4C";
outerbackcolor = "#142733";
fontcolor = "white"

favorite_cmaps = ['cool', 'autumn', 'autumn_r', 'Set2_r', 'cool_r',
                  'gist_rainbow', 'prism', 'rainbow', 'spring']


# FUNCTIONS: d, p, sp, table_of_contents, display_me, sample_df, see,
#	     list_to_table, div_print, overview, missing_values, fancy_plot

# .......................IMPORTS....................................... #
def pd_np_mpl_import():
    global pd
    global np
    global plt
    global reload
    global sns

    pd = __import__('pandas', globals(), locals())
    np = __import__('numpy', globals(), locals())
    matplotlib = __import__('matplotlib', globals(), locals())
    plt = matplotlib.pyplot
    sns = __import__('seaborn', globals(), locals())
    importlib = __import__('importlib', globals(), locals())
    reload = importlib.reload


def url_import():
    global urlretrieve
    urllib = __import__('urllib', globals(), locals())
    urlretrieve = urllib.request.urlretrieve

def yf_import():
    global yf
    yf = __import__('yfinance', globals(), locals())

def import_all():
    pd_np_mpl_import()
    url_import()
    yf_import()


# ......................TINY_GUYS....................................... #

def sp(): print('');

def p(x): print(x); sp()

def d(x): display(x); sp()

# .......................Complementary Colors....................................... #
def get_complementary(color):
    color = color[1:]
    color = int(color, 16)
    comp_color = 0xFFFFFF ^ color
    comp_color = "#%06X" % comp_color
    return comp_color

# .......................Pretty....................................... #

def pretty(data, label=None, fontsize=3,
            bgcolor=pretty_background,
            textcolor=pretty_text, width=None
            ):
    from IPython.display import HTML
    import numpy as np

    if isinstance(data, np.ndarray):
        data = list(data)

    if label:
        output_df = pd.DataFrame([label, data])
    else:
        output_df = pd.DataFrame([[data]])

    if label:
        df_styler = (
            [{'selector': '.row0',
              'props': [('background-color', pretty_label_background),
                        ('color', pretty_label_text),
                        ('font-size', '15px'),
                        ('font-weight', 550),
                        ('text-align', 'left'),
                        ('padding', '3px 5px 3px 5px')]},
             {'selector': '.row1',
              'props': [('background-color', pretty_background),
                        ('color', pretty_text),
                        ('font-size', '15px'),
                        ('font-weight', 'bold'),
                        ('text-align', 'left'),
                        ('padding', '3px 5px 5px 5px')]},
             {'selector': 'tbody',
              'props': [('border', '1px solid'),
                        ('border-color', 'black')]},
             {'selector': 'tr',
              'props': [('border', '0.8px solid'),
                        ('border-color', 'black')]}])
    else:
        df_styler = (
            [{'selector': '.row0',
              'props': [('background-color', pretty_background),
                        ('color', pretty_text),
                        ('font-size', '15px'),
                        ('font-weight', 'bold'),
                        ('text-align', 'left'),
                        ('padding', '3px 2px 5px 5px')]},
             {'selector': 'tbody',
              'props': [('border', '1px solid'),
                        ('border-color', 'black')]},
             {'selector': 'tr',
              'props': [('border', '0.8px solid'),
                        ('border-color', 'black')]}])

    display(output_df.style.hide(axis='index') \
            .hide(axis='columns') \
            .set_table_styles(df_styler))
    sp()

# .......................Div Print....................................... #
def div_print(text, width='auto', bgcolor=bgcolor, text_color=text_color,
              fontsize=2
              ):
    from IPython.display import HTML as html_print

    if width == 'auto':
        font_calc = {6: 2.75, 5: 2.5, 4: 2.5, 3: 3, 2: 4}
        width = str(len(text) * fontsize * font_calc[fontsize]) + "px"

    else:
        if type(width) != str:
            width = str(width)
        if width[-1] == "x":
            width = width
        elif width[-1] != '%':
            width = width + "px"

    return display(html_print("<span style = 'display: block; width: {}; \
						line-height: 2; background: {};\
						margin-left: auto; margin-right: auto;\
						border: 1px solid text_color;\
						border-radius: 3px; text-align: center;\
						padding: 3px 8px 3px 8px;'>\
						<b><font size={}><text style=color:{}>{}\
						</text></font></b></style>".format(width, bgcolor,
                                                           fontsize,
                                                           text_color, text)))

# .......................DESCRIBE_EM....................................... #
def describe_em(df, col_list):
	df_list = []
	for column in col_list:
		df_tuple = (df[column].describe(), 'df.' + column)
		df_list.append(df_tuple)
	multi(df_list)


# .......................Time Stamp Converter....................................... #
# Write a function to convert any column in a df that is a timestamp
# to date, hour, and min only
# Find columns that dtype is timestamp
def time_stamp_converter(df):

    def find_datetime(df):
        datetime_cols = []

        for col in df.columns:
            if df[col].dtype.name.startswith('datetime64'):
                datetime_cols.append(col)
        return datetime_cols

    def find_timestamp(df):
        timestamp_cols = []
        for col in df.columns:
            if isinstance(df[col][0], pd._libs.tslibs.timestamps.Timestamp):
                timestamp_cols.append(col)
        return timestamp_cols

    datetime_cols = find_datetime(df)
    for col in datetime_cols:
        df[col] = df[col].dt.strftime('%Y-%m-%d')
    return df

    timestamp_cols = find_timestamp(df)
    for col in timestamp_cols:
        df[col] = df[col].dt.strftime('%Y-%m-%d')
    return df


# .......................DISPLAY_ME........................................ #
def head_tail_vert(df, num, title, bgcolor=bgcolor,
                    text_color=text_color, fontsize=4,
                    intraday=False):
    from IPython.core.display import HTML

    if type(df) != pd.core.frame.DataFrame:
        df = df.copy().to_frame()

    if not intraday:
        df = time_stamp_converter(df.copy())
        if df.index.dtype.name.startswith('datetime64'):
            df.index = df.index.strftime('%Y-%m-%d')
        elif isinstance(df.index[0], pd._libs.tslibs.timestamps.Timestamp):
            df.index = df.index.strftime('%Y-%m-%d')

    head_data = "<center>" + df.head(num).to_html()
    tail_data = "<center>" + df.tail(num).to_html()

    print("")
    div_print(f'{title}: head({num})', fontsize=fontsize,
              bgcolor=bgcolor, text_color=text_color)
    display(HTML(head_data))
    print("")
    div_print(f'{title}: tail({num})', fontsize=fontsize,
              bgcolor=bgcolor, text_color=text_color)
    display(HTML(tail_data))
    print("")

def head_tail_horz(df, num, title, bgcolor=bgcolor,
                   text_color=text_color, precision=2,
                   intraday=False, title_fontsize=4,
                   table_fontsize="12px"):

    if type(df) != pd.core.frame.DataFrame:
        df = pd.DataFrame(df.copy())

    if not intraday:
        df = time_stamp_converter(df.copy())
        if df.index.dtype.name.startswith('datetime64'):
            df.index = df.index.strftime('%Y-%m-%d')
        elif isinstance(df.index[0], pd._libs.tslibs.timestamps.Timestamp):
            df.index = df.index.strftime('%Y-%m-%d')

    div_print(f'{title}', fontsize=title_fontsize,
              bgcolor=bgcolor, text_color=text_color)
    multi([(df.head(num),f"head({num})"),
           (df.tail(num),f"tail({num})")],
          fontsize=table_fontsize, precision=precision,
          intraday=intraday)

# .......................SEE....................................... #

def see(data, title=None, width="auto", fontsize=4,
        bgcolor=bgcolor, text_color=text_color,
        intraday=False):

    if title != None:
        div_print(f"{title}", fontsize=fontsize, width=width,
                  bgcolor=bgcolor, text_color=text_color)

    if isinstance(data, pd.core.frame.DataFrame):
        if not intraday:
            data = time_stamp_converter(data.copy())
            if data.index.dtype.name.startswith('datetime64'):
                data.index = data.index.strftime('%Y-%m-%d')
            elif isinstance(data.index[0], pd._libs.tslibs.timestamps.Timestamp):
                data.index = data.index.strftime('%Y-%m-%d')

        display(HTML("<center>" + data.to_html()));
        sp()
    elif isinstance(data, pd.core.series.Series):
        if data.index.dtype.name.startswith('datetime64'):
            data.index = data.index.strftime('%Y-%m-%d')
        elif isinstance(data.index[0], pd._libs.tslibs.timestamps.Timestamp):
            data.index = data.index.strftime('%Y-%m-%d')

        display(HTML("<center>" + data.to_frame().to_html()));
        sp()
    else:
        try:
            display(HTML("<center>" + data.to_frame().to_html()));
            sp()
        except:
            pretty(data, title);
            sp()


# .......................FORCE_DF....................................... #
def date_only(data, intraday=False):
    if intraday == False:
        if data.index.dtype == 'datetime64[ns]':
            data.index = data.index.strftime('%Y-%m-%d')
            return data
        elif isinstance(data.index[0], pd._libs.tslibs.timestamps.Timestamp):
            data.index = data.index.strftime('%Y-%m-%d')
            return data
        else:
            return data
    else:
        return data


def force_df(data, intraday=False):
    if isinstance(data, pd.core.series.Series):
        return date_only(data, intraday=intraday).to_frame()
    elif isinstance(data, pd.core.frame.DataFrame):
        return date_only(data, intraday=intraday)
    else:
        try:
            return pd.Series(data).to_frame()
        except:
            return div_print("The data cannot be displayed.")

# .......................MULTI....................................... #

def multi(data_list, fontsize='15px', precision=2, intraday=False):
    from IPython.display import display_html

    caption_style = [{
        'selector': 'caption',
        'props': [
            ('background', bgcolor),
            ('border-radius', '3px'),
            ('padding', '5px'),
            ('color', text_color),
            ('font-size', fontsize),
            ('font-weight', 'bold')]}]

    thousands = ",";
    spaces = "&nbsp;&nbsp;&nbsp;"
    table_styling = caption_style

    stylers = []
    for idx, pair in enumerate(data_list):
        if len(pair) == 2:
            table_attribute_string = "style='display:inline-block'"
        elif pair[2] == 'center':
            table_attribute_string = "style='display:inline-grid'"
        styler = force_df(data_list[idx][0], intraday=intraday).style \
            .set_caption(data_list[idx][1]) \
            .set_table_attributes(table_attribute_string) \
            .set_table_styles(table_styling).format(precision=precision,
                                                    thousands=thousands)
        stylers.append(styler)

    if len(stylers) == 1:
        display_html('<center>' + stylers[0]._repr_html_(), raw=True); sp();
    elif len(stylers) == 2:
        display_html('<center>' + stylers[0]._repr_html_() + spaces + stylers[1]._repr_html_() + spaces, raw=True); sp();
    elif len(stylers) == 3:
        display_html('<center>' + stylers[0]._repr_html_() + spaces + stylers[1]._repr_html_() + spaces + stylers[
            2]._repr_html_() + spaces, raw=True); sp();
    elif len(stylers) == 4:
        display_html('<center>' + stylers[0]._repr_html_() + spaces + stylers[1]._repr_html_() + spaces + stylers[
            2]._repr_html_() + spaces + stylers[3]._repr_html_() + spaces, raw=True); sp();

# .......................LIST_TO_TABLE....................................... #

def list_to_table(display_list, num_cols, title, width="auto",
                  bgcolor=bgcolor, text_color=text_color
                  ):
    div_print(f"{title}", fontsize=4, width=width,
              bgcolor=bgcolor, text_color=text_color)

    count = 0
    current = '<center><table><tr>'
    length = len(display_list)
    num_rows = round(length / num_cols) + 1

    for h in range(num_rows):
        for i in range(num_cols):
            try:
                current += ('<td>' + display_list[count] + '</td>')
            except IndexError:
                current += '<td>' + ' ' + '</td>'
            count += 1
        current += '</tr><tr>'
    current += '</tr></table></center>'
    display(HTML(current))

# .......................MISSING_VALUES....................................... #

def missing_values(df, bgcolor=bgcolor, text_color=text_color):
    from IPython.display import HTML
    pd.options.display.float_format = '{:,.0f}'.format
    missing_log = []
    for column in df.columns:
        missing_values = df[column].isna().sum()
        missing_log.append([column, missing_values])
    missing = pd.DataFrame(missing_log, columns=['column name', 'missing'])
    div_print(f'Columns and Missing Values', fontsize=3, width="38%",
              bgcolor=bgcolor, text_color=text_color)
    missing = "<center>" + missing.to_html()
    display(HTML(missing))


# ............................FANCY_PLOT....................................... #

def fancy_plot(data, kind="line", title=None, legend_loc='upper right',
               xlabel=None, ylabel=None, logy=False, outerbackcolor=outerbackcolor,
               innerbackcolor=innerbackcolor, fontcolor=fontcolor, cmap='cool',
               label_rot=None):

    import random
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    if isinstance(data, pd.core.series.Series):
        data = data.to_frame()

    mpl.rcParams['xtick.color'] = outerbackcolor
    mpl.rcParams['ytick.color'] = outerbackcolor
    mpl.rcParams['font.family'] = 'monospace'
    fig = plt.subplots(facecolor=outerbackcolor, figsize=(13, 7))
    ax = plt.axes();
    if kind == 'line':
        data.plot(kind='line', ax=ax, rot=label_rot, cmap=cmap, logy=logy)
    else:
        data.plot(kind=kind, ax=ax, rot=label_rot, cmap=cmap, logy=logy);
    plt.style.use("ggplot");
    ax.set_facecolor(innerbackcolor)
    ax.grid(color=fontcolor, linestyle=':', linewidth=0.75, alpha=0.75)
    plt.tick_params(labelrotation=40);
    plt.title(title, fontsize=23, pad=20, color=fontcolor);
    plt.ylabel(ylabel, fontsize=18, color=fontcolor);
    plt.xlabel(xlabel, fontsize=18, color=fontcolor);
    plt.xticks(fontsize=10, color=fontcolor)
    plt.yticks(fontsize=10, color=fontcolor)
    if legend_loc is None:
        ax.get_legend().remove()
    else:
        plt.legend(labels=data.columns, fontsize=15, loc=legend_loc,
                   facecolor=outerbackcolor, labelcolor=fontcolor)


# ************************************************************************** #
# ****************************TIME SERIES ********************************** #


# ****************************Featurize Datetime Index********************************** #
def featurize_datetime_index(df, daytime=True):
    '''
    Create time series features based on a datetime index
    '''

    df = df.copy()

    df['hour'] = df.index.hour
    df['weekday'] = df.index.dayofweek
    df['weekday_name'] = df.index.strftime('%A')
    df['month'] = df.index.month
    df['month_name'] = df.index.strftime('%B')
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
    df['week_of_year'] = df.index.isocalendar().week
    df['day_of_year'] = df.index.dayofyear

    if daytime:
        # Add column with category for time of day:
        # midnight, early_morning, late_morning, afternoon, evening, night
        def time_of_day(hour):
            if hour >= 0 and hour < 6:
                return 'midnight'
            elif hour >= 6 and hour < 9:
                return 'early_morning'
            elif hour >= 9 and hour < 12:
                return 'late_morning'
            elif hour >= 12 and hour < 15:
                return 'afternoon'
            elif hour >= 15 and hour < 18:
                return 'evening'
            else:
                return 'night'

                df['time_of_day'] = (df['hour'].apply(time_of_day)).astype('category')

        df['weekday_name'] = df['weekday_name'].astype('category')
        df['month_name'] = df['month_name'].astype('category')
        df['week_of_year'] = df.week_of_year.astype(float)

    return df


# ****************************Add Change Column ********************************** #

def add_change_column(df, column_changing, new_col_name):
	df['previous'] = df[column_changing].shift()
	df = df.drop(df.index[0])
	df[new_col_name] = df[column_changing] - df.previous
	df = df.drop(columns = ['previous'])
	return df


# ****************************Add Year Lags ********************************** #
def year_lags(df, target_column, lag_label_list):
    target_map = df[target_column].to_dict()
    inputs = lag_label_list.copy()

    for tup in inputs:
        df[tup[1]] = (df.index - pd.Timedelta(tup[0])).map(target_map)

    return df

# ****************************Add Accuracy********************************** #
def get_accuracy(df, pred_col, actual_col):
    from sklearn.metrics import mean_squared_error
    df = df.copy()

    df['abs_acc'] = (1 - (abs(df[actual_col] -
                              df[pred_col]) / df[actual_col])) * 100

    range_diff = np.max(df[actual_col]) - np.min(df[actual_col])

    df['rel_acc'] = (1 - (abs(df[actual_col] -
                              df[pred_col]) / range_diff)) * 100

    range_std = np.std(df[actual_col])

    df['sharpe'] = (abs(df[actual_col] -
                        df[pred_col]) / range_std)

    rmse = np.sqrt(mean_squared_error(df[actual_col],
                                      df[pred_col]))

    div_print(f"Average RMSE: {rmse:,.2f}  |  Average sharpe ratio: {df['sharpe'].mean():.2f} ", fontsize=3)
    div_print(
        f"Average absolute accuracy: {df['abs_acc'].mean():.2f}%  |  Average relative accuracy: {df['rel_acc'].mean():.2f}% ",
        fontsize=3)

    return df

# **************************** BoxPlot Correlation ********************************** #
def boxplot_correlation(df, feature_x, feature_y, order=None, palette=None):
    fig, ax = plt.subplots(figsize=(13, 7), facecolor='outerbackcolor')
    ax.set_facecolor(innerbackcolor)

    sns.boxplot(data=df,
                x=feature_x,
                y=feature_y,
                order=order,
                palette=palette)

    x_name = str(df[feature_x].name)
    y_name = str(df[feature_y].name)

    ax.grid()
    plt.xlabel(x_name, color='white', fontsize=15)
    plt.ylabel(y_name, color='white', fontsize=15)
    plt.xticks(color='white');
    plt.yticks(color='white');
    plt.title(f'Feature Correlation: {x_name.capitalize()} - {y_name.capitalize()}',
              fontsize=20, pad=20, color='white');

# **************************** Get Daily Error ********************************** #
def get_daily_error(df, actual_col, pred_col, num_examples,
                    ascending=False
                    ):
    temp = df[[actual_col, pred_col]].copy()
    temp['date'] = temp.index.strftime('%A, %b %d, %Y')
    temp['error'] = np.abs(df[actual_col] - df[pred_col])

    results = temp.sort_values("error", ascending=ascending)

    error_style = {'error': [{'selector': '',
                              'props': [('color', 'red'),
                                        ('font-weight', 'bold'),
                                        ('padding-right', '15px'),
                                        ('padding-left', '15px')]}],
                   'date': [{'selector': 'td',
                             'props': [('color', 'blue'),
                                       ('font-weight', 'bold'),
                                       ('padding-right', '15px'),
                                       ('padding-left', '15px')]}],
                   'prediction': [{'selector': 'td',
                                   'props': [('padding-right', '25px'),
                                             ('padding-left', '15px')]}]}

    if ascending == True:
        pretty(f'Daily error for the {num_examples} days with the lowest error:',
               fontsize=4)
    else:
        pretty(f'Daily error for the {num_examples} days with the highest error:',
               fontsize=4)

    return results[['date',
                    'error',
                    pred_col,
                    actual_col]].head(num_examples).style.hide(axis='index') \
        .set_table_styles(error_style) \
        .format(precision=3, thousands=",")