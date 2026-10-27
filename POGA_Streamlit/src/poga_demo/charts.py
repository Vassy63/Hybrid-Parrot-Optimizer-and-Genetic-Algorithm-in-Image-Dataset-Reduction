"""Charts shared by the demo and the archived experiments."""
import altair as alt
import pandas as pd


def class_distribution_chart(manifest):
    frame = pd.DataFrame(manifest)
    counts = frame.groupby('class')['selected'].agg(['size', 'sum']).rename(
        columns={'size': 'Trước rút gọn', 'sum': 'Sau rút gọn'})
    stages = ['Trước rút gọn', 'Sau rút gọn']
    data = counts.reset_index().melt('class', var_name='stage', value_name='images')
    base = alt.Chart(data).encode(
        x=alt.X('class:N', title='Lớp ảnh', axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset('stage:N', sort=stages),
        y=alt.Y('images:Q', title='Số ảnh', scale=alt.Scale(zero=True, padding=25)),
        color=alt.Color('stage:N', sort=stages, title=None,
                        scale=alt.Scale(domain=stages, range=['#4c8bd6', '#2ca58d']),
                        legend=alt.Legend(orient='top')),
        tooltip=[alt.Tooltip('class:N', title='Lớp'), alt.Tooltip('stage:N', title='Tập'),
                 alt.Tooltip('images:Q', title='Số ảnh', format=',.0f')])
    return (base.mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3) +
            base.mark_text(dy=-9, fontSize=12).encode(
                text=alt.Text('images:Q', format=',.0f'))).properties(height=310)


def comparison_chart(frame, metric, title, *, error=None, number_format='.2f'):
    data = frame.copy()
    if error:
        data['lower'] = data[metric] - data[error]
        data['upper'] = data[metric] + data[error]
    base = alt.Chart(data).encode(
        y=alt.Y('Method:N', title=None,
                sort=data.sort_values(metric, ascending=False)['Method'].tolist(),
                axis=alt.Axis(labelLimit=190)),
        x=alt.X(f'{metric}:Q', title=title, scale=alt.Scale(zero=True, padding=45)),
        tooltip=[alt.Tooltip('Method:N', title='Phương pháp'),
                 alt.Tooltip(f'{metric}:Q', title=title, format=number_format)] +
                ([alt.Tooltip(f'{error}:Q', title='Độ lệch chuẩn', format='.2f')] if error else []))
    bars = base.mark_bar(color='#4c8bd6', cornerRadiusEnd=3)
    labels = base.mark_text(align='left', dx=6, fontSize=12, color='#4c8bd6').encode(
        text=alt.Text(f'{metric}:Q', format=number_format))
    if error:
        intervals = base.mark_rule(color='#2ca58d', strokeWidth=2).encode(
            x='lower:Q', x2='upper:Q')
        labels = labels.encode(x='upper:Q')
        return (bars + intervals + labels).properties(height=max(180, len(data) * 30))
    return (bars + labels).properties(height=max(180, len(data) * 30))
