"""Run with: python -m streamlit run src/app.py"""
from pathlib import Path
import io
import sys
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import streamlit as st

from poga_demo.jobs import JobManager, TERMINAL, read_events, read_json
from poga_demo.settings import DemoSettings, METHODS, MODELS, PROJECT_ROOT
from poga_demo.results import read_results, summarize
from poga_demo.charts import class_distribution_chart, comparison_chart

st.set_page_config(page_title='Demo PO-GA', page_icon=':material/analytics:', layout='wide')
st.markdown('<style>' + (PROJECT_ROOT / 'assets/demo.css').read_text(encoding='utf-8') + '</style>',
            unsafe_allow_html=True)


@st.cache_resource
def manager():
    return JobManager()


def load_table(path):
    return pd.read_csv(path) if Path(path).is_file() else pd.DataFrame()


def elapsed_text(seconds):
    seconds = int(seconds or 0)
    return f'{seconds // 60:02d}:{seconds % 60:02d}'


def show_run_results(folder):
    summary = read_json(folder / 'summary.json')
    if not summary:
        return
    st.subheader('Kết quả lượt demo')
    a, b, c, d = st.columns(4)
    a.metric('Ảnh trước → sau', f"{summary['train_before']:,} → {summary['train_after']:,}")
    b.metric('Rút gọn thực tế', f"{summary['actual_dr']:.2f}%")
    c.metric('Fitness', f"{summary['fitness']:.5f}")
    last = summary['rows'][-1]
    d.metric('Test Accuracy', f"{last['Test_Acc']:.2f}%")
    st.dataframe(pd.DataFrame(summary['rows']).rename(columns={
        'Method':'Phương pháp','Model':'CNN','Subset_Size_M':'Ảnh train','DR_Percent':'DR (%)',
        'Test_Acc':'Test Acc (%)','Macro_F1':'Macro F1 (%)','ARR_Percent':'ARR (%)',
        'Train_Time_Sec':'Thời gian train (s)','Speedup':'Tăng tốc (lần)'}), hide_index=True, width='stretch')
    performance = pd.DataFrame(summary['rows'])
    left, right = st.columns(2)
    with left:
        st.markdown('**Độ chính xác trên tập test**')
        st.altair_chart(comparison_chart(performance, 'Test_Acc', 'Test Accuracy (%)'), width='stretch')
    with right:
        st.markdown('**Thời gian huấn luyện CNN**')
        st.altair_chart(comparison_chart(performance, 'Train_Time_Sec', 'Thời gian (giây)'), width='stretch')
    chart = load_table(folder / 'convergence_histories.csv')
    if not chart.empty:
        left, right = st.columns(2)
        left.markdown('**Fitness theo vòng tối ưu**')
        left.line_chart(chart.set_index('Iteration')[['total']], width='stretch')
        right.markdown('**Các thành phần mục tiêu**')
        right.line_chart(chart.set_index('Iteration')[['div','cov','bal','com']], width='stretch')
        right.caption('Khi bật tỷ lệ mục tiêu, Com đo mức bám mục tiêu, khác với DR thực tế.')
    curves = []
    for label, key in [('B0','b0'), (last['Method'], read_json(folder / 'config.json').get('settings',{}).get('method','s1'))]:
        path = folder / 'model_runs' / key / 'logs' / 'training_log.csv'
        if path.exists():
            frame = load_table(path)
            frame['Phương pháp'] = label
            curves.append(frame)
    if curves:
        learning = pd.concat(curves, ignore_index=True)
        left, right = st.columns(2)
        with left:
            st.markdown('**Accuracy trên validation theo epoch**')
            st.line_chart(learning.assign(val_acc=learning['val_acc'] * 100),
                x='epoch', y='val_acc', color='Phương pháp',
                x_label='Epoch', y_label='Validation Accuracy (%)', width='stretch')
        with right:
            st.markdown('**Loss trên validation theo epoch**')
            st.line_chart(learning, x='epoch', y='val_loss', color='Phương pháp',
                x_label='Epoch', y_label='Validation loss', width='stretch')
        with st.expander('Nhật ký học theo epoch'):
            st.dataframe(learning, hide_index=True, width='stretch')
    manifest = read_json(folder / 'image_manifest.json', [])
    if manifest:
        st.markdown('**Phân bố lớp trước và sau rút gọn**')
        st.altair_chart(class_distribution_chart(manifest), width='stretch')
        for title, chosen in [('Một số ảnh được giữ lại', True), ('Một số ảnh bị loại', False)]:
            with st.expander(title):
                examples = [r for r in manifest if r['selected'] == chosen][:12]
                if not examples:
                    st.caption('Không có ảnh trong nhóm này.')
                for start in range(0, len(examples), 4):
                    for column, row in zip(st.columns(4), examples[start:start+4]):
                        if Path(row['path']).is_file():
                            column.image(row['path'], caption=row['class'], width='stretch')
                        else:
                            column.caption('Ảnh nguồn đã được di chuyển.')
    st.download_button('Tải bảng kết quả CSV', (folder/'experiment_results.csv').read_bytes(),
        file_name=f'{folder.name}.csv', mime='text/csv', key='csv_'+folder.name)
    if st.button('Chuẩn bị ZIP toàn bộ lượt chạy', key='zip_'+folder.name):
        data = io.BytesIO()
        with zipfile.ZipFile(data, 'w', zipfile.ZIP_DEFLATED) as archive:
            for file in folder.rglob('*'):
                if file.is_file() and not file.name.endswith('.tmp'):
                    archive.write(file, arcname=str(file.relative_to(folder)))
        st.download_button('Tải ZIP kết quả, log và checkpoint', data.getvalue(), file_name=folder.name+'.zip', key='dlzip_'+folder.name)


@st.fragment(run_every=2.0)
def run_panel():
    chosen = st.session_state.get('run_folder')
    if not chosen:
        st.info('Chọn cấu hình ở bên trái rồi bấm “Chạy demo”.')
        return
    folder = Path(chosen)
    state = read_json(folder / 'status.json')
    status = state.get('state','starting')
    labels = {'starting':'Khởi động','running':'Đang chạy','succeeded':'Hoàn tất',
              'failed':'Chạy lỗi','cancelled':'Đã dừng','time_limit':'Hết ngân sách thời gian'}
    a, b = st.columns([3,1])
    a.metric('Trạng thái', labels.get(status,status), help=str(state.get('message','Đang chuẩn bị')))
    elapsed = max(0, time.time()-state['started_at']) if status not in TERMINAL and 'started_at' in state else state.get('elapsed',0)
    b.metric('Đã chạy', elapsed_text(elapsed))
    cfg = read_json(folder / 'config.json').get('settings',{})
    limit = cfg.get('time_limit_minutes',15)*60
    if status not in TERMINAL:
        st.progress(min(1.0, elapsed/limit), text='Thời gian chạy')
    if status not in TERMINAL and manager().active():
        if st.button('Dừng lượt đang chạy', type='secondary', key='cancel_run'):
            manager().cancel()
            st.rerun(scope='fragment')
    events = read_events(folder)
    fitness = [{'Đánh giá nghiệm':e['evaluations'],'Fitness tốt nhất':e['fitness']} for e in events if 'evaluations' in e]
    if fitness and status not in TERMINAL:
        st.line_chart(pd.DataFrame(fitness).drop_duplicates('Đánh giá nghiệm').set_index('Đánh giá nghiệm'))
    if status == 'succeeded' and (folder/'summary.json').exists():
        show_run_results(folder)
    elif status in ['failed','time_limit','cancelled']:
        st.warning(state.get('message','Phiên chưa hoàn tất.'))
    elif status == 'succeeded' and (folder/'environment.json').exists():
        info = read_json(folder/'environment.json')
        if info.get('cuda_available'):
            st.success('GPU sẵn sàng: '+str(info.get('gpu')))
        else:
            st.warning('PyTorch chưa nhận GPU CUDA. Có thể chạy CPU hoặc cài bản PyTorch CUDA trước khi đo thời gian demo.')
        missing=[name for name,ready in info.get('dependencies',{}).items() if not ready]
        if missing:
            st.error('Thiếu thư viện: '+', '.join(missing)+'. Chạy setup_demo.bat trước.')
        with st.expander('Chi tiết môi trường Python'):
            st.json(info)
    with st.expander('Tiến độ chi tiết và log'):
        if events:
            st.dataframe(pd.DataFrame(events).tail(30), hide_index=True, width='stretch')
        log = folder / 'console.log'
        if log.is_file():
            with log.open('rb') as handle:
                size = handle.seek(0,2);handle.seek(max(0,size-16000));text=handle.read().decode('utf-8',errors='replace')
            st.code(text, language='text')


def result_archive():
    st.subheader('Kết quả thực nghiệm 30 seeds')
    st.caption('Các số liệu đã huấn luyện trước được tách khỏi lượt demo vừa chạy.')
    default = PROJECT_ROOT/'assets/results_30seeds/experiment_results_30seeds.csv'
    try:
        rows = read_results(default)
        agg = pd.DataFrame(summarize(rows))
    except (OSError, UnicodeError, ValueError) as exc:
        st.warning(str(exc));return
    raw = pd.DataFrame(rows)
    a,b,c = st.columns(3)
    a.metric('Bản ghi',len(raw));b.metric('Phương pháp',raw['Method'].nunique());c.metric('Seeds',raw['Seed'].nunique())
    model = st.selectbox('Kiến trúc CNN',list(agg['Model'].unique()),format_func=MODELS.get,key='archive_model')
    filtered = agg[agg['Model']==model].sort_values('Test_Acc_mean',ascending=False)
    left, right = st.columns(2)
    with left:
        st.markdown('**Test Accuracy trung bình ± độ lệch chuẩn**')
        st.altair_chart(comparison_chart(filtered, 'Test_Acc_mean', 'Test Accuracy (%)',
            error='Test_Acc_std'), width='stretch')
    with right:
        st.markdown('**Macro F1 trung bình ± độ lệch chuẩn**')
        st.altair_chart(comparison_chart(filtered, 'Macro_F1_mean', 'Macro F1 (%)',
            error='Macro_F1_std'), width='stretch')
    left, right = st.columns(2)
    with left:
        st.markdown('**Tỷ lệ rút gọn trung bình**')
        st.altair_chart(comparison_chart(filtered, 'DR_Percent_mean', 'DR (%)'), width='stretch')
    with right:
        st.markdown('**Tăng tốc huấn luyện trung bình**')
        st.altair_chart(comparison_chart(filtered, 'Speedup_mean', 'Speedup (lần)'), width='stretch')
    columns = [c for c in ['Method','Count','Test_Acc_mean','Test_Acc_std','Macro_F1_mean','DR_Percent_mean','ARR_Percent_mean','Speedup_mean'] if c in filtered]
    st.dataframe(filtered[columns], hide_index=True,width='stretch')
    st.caption('Mean: trung bình; std: độ lệch chuẩn mẫu qua các seed. Đây là thứ hạng theo trung bình quan sát, không phải kết luận về ý nghĩa thống kê.')
    with st.expander('Dữ liệu từng lần chạy'):
        st.dataframe(raw,width='stretch',hide_index=True)


with st.container(key='demo_hero'):
    st.markdown('<div class="hero-eyebrow">TỐI ƯU DỮ LIỆU ẢNH</div>', unsafe_allow_html=True)
    st.title('Demo PO-GA')
    st.write('Rút gọn tập ảnh, huấn luyện CNN và khám phá kết quả thực nghiệm.')
    st.markdown('<div class="hero-tags"><span>Genetic Algorithm</span>'
                '<span>Parrot Optimizer</span><span>Image Classification</span></div>',
                unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-brand"><div class="brand-mark">PG</div>'
                '<div>Demo PO-GA<small>Không gian thực nghiệm</small></div></div>',
                unsafe_allow_html=True)
    st.subheader('Thiết lập demo')
    st.session_state.setdefault('samples', DemoSettings.num_samples)
    st.session_state.setdefault('epochs', DemoSettings.epochs)
    method = st.selectbox('Phương pháp rút gọn', list(METHODS),format_func=METHODS.get)
    model = st.selectbox('CNN đánh giá',list(MODELS),format_func=MODELS.get)
    samples = st.number_input('Số ảnh train trước rút gọn',min_value=120,max_value=11228,step=120,key='samples')
    dr = st.slider('Rút gọn tối đa (%)',0,70,70,5)
    epochs = st.number_input('Epochs',min_value=1,max_value=30,step=1,key='epochs')
    baseline = st.checkbox('Huấn luyện thêm B0 để so sánh',value=True)
    with st.expander('Tham số nâng cao'):
        population=st.number_input('Quần thể',min_value=6,max_value=40,value=10)
        iterations=st.number_input('Tổng vòng tối ưu',min_value=2,max_value=200,value=20)
        phase=st.number_input('Độ dài mỗi pha',min_value=1,max_value=100,value=5)
        seed=st.number_input('Seed tối ưu và CNN',min_value=0,max_value=2**32-1,value=42)
        batch=st.selectbox('Batch size',[16,32,64,128],index=2)
        device=st.selectbox('Thiết bị',['auto','cuda','cpu'])
    cfg=DemoSettings(method=method,model=model,
        num_samples=int(samples),reduction_percent=dr,population_size=int(population),iterations=int(iterations),
        phase_size=int(phase),epochs=int(epochs),batch_size=batch,seed=int(seed),time_limit_minutes=15,
        run_baseline=baseline,device=device)
    if st.button('Chạy demo',type='primary',width='stretch',disabled=manager().active()):
        try:
            folder=manager().start(cfg)
            st.session_state['run_folder']=str(folder)
        except (ValueError,OSError) as exc:
            st.error(str(exc))
    if st.button('Kiểm tra môi trường',width='stretch',disabled=manager().active()):
        try:
            st.session_state['run_folder']=str(manager().start(cfg,mode='diagnose'))
        except (ValueError,OSError) as exc:
            st.error(str(exc))

demo_tab, archive_tab, history_tab = st.tabs(['Chạy demo','Kết quả 30 seeds','Lịch sử'])
with demo_tab:
    run_panel()
with archive_tab:
    result_archive()
with history_tab:
    folders=manager().folders()
    if not folders:
        st.info('Chưa có lượt chạy.')
    else:
        items=[]
        for folder in folders:
            state=read_json(folder/'status.json')
            config=read_json(folder/'config.json').get('settings',{})
            items.append({'Phiên':folder.name,'Trạng thái':state.get('state'),'Phương pháp':config.get('method'),
                'CNN':config.get('model'),'Ảnh':config.get('num_samples'),'DR tối đa (%)':config.get('reduction_percent')})
        st.dataframe(pd.DataFrame(items),hide_index=True,width='stretch')
        chosen=st.selectbox('Mở một phiên',folders,format_func=lambda p:p.name)
        if st.button('Xem phiên này'):
            st.session_state['run_folder']=str(chosen)
            st.rerun()
