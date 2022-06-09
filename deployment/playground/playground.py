import base64
import os
from copy import deepcopy
from urllib.request import urlopen

import av
import numpy as np
import requests
import streamlit as st
from docarray import Document, DocumentArray
from docarray import __version__ as docarray_version
from streamlit_webrtc import ClientSettings, webrtc_streamer

WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)


root_data_dir = (
    'https://storage.googleapis.com/jina-fashion-data/data/one-line/datasets/'
)

ds_set = [
    'nft-monkey',
    'deepfashion',
    'nih-chest-xrays',
    'stanford-cars',
    'bird-species',
    'best-artworks',
    'geolocation-geoguessr',
    'rock-lyrics',
    'pop-lyrics',
    'rap-lyrics',
    'indie-lyrics',
    'metal-lyrics',
]

SURVEY_LINK = 'https://10sw1tcpld4.typeform.com/to/VTAyYRpR?utm_source=cli'


def deploy_streamlit():
    """
    We want to provide the end-to-end experience to the user.
    Please deploy a streamlit playground on k8s/local to access the api.
    You can get the starting point for the streamlit application from alex.
    """
    # Header
    # put this on the top so that it shows immediately, while the rest is loading
    st.set_page_config(page_title="NOW", page_icon='https://jina.ai/favicon.ico')
    _, mid, _ = st.columns([0.8, 1, 1])
    with open('./logo.svg', 'r') as f:
        svg = f.read()
    with mid:
        b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
        html = r'<img width="250" src="data:image/svg+xml;base64,%s"/>' % b64
        st.write(html, unsafe_allow_html=True)
    setup_session_state()
    query_parameters = st.experimental_get_query_params()
    print(f"Received query params: {query_parameters}")
    HOST = query_parameters.get('host')[0]
    PORT = (
        query_parameters.get('port')[0] if 'port' in query_parameters.keys() else None
    )
    OUTPUT_MODALITY = query_parameters.get('output_modality')[0]
    INPUT_MODALITY = query_parameters.get('input_modality')[0]
    DATA = (
        query_parameters.get('data')[0] if 'data' in query_parameters.keys() else None
    )
    # TODO: fix such that can call 'localhost' instead of 'jinanowtesting'
    if HOST == 'gateway':  # need to call now-bff as we communicate between pods
        URL_HOST = f"http://now-bff/api/v1/{OUTPUT_MODALITY}/search"
    else:
        URL_HOST = f"https://nowrun.jina.ai/api/v1/{OUTPUT_MODALITY}/search"
        # URL_HOST = f"localhost/api/v1/{OUTPUT_MODALITY}/search"

    da_img = None
    da_txt = None

    # General
    TOP_K = 9
    DEBUG = os.getenv("DEBUG", False)
    DATA_DIR = "../data/images/"

    if DATA in ds_set:
        if OUTPUT_MODALITY == 'image':
            output_modality_dir = 'jpeg'
            data_dir = root_data_dir + output_modality_dir + '/'
            da_img, da_txt = load_data(
                data_dir + DATA + f'.img10-{docarray_version}.bin'
            ), load_data(data_dir + DATA + f'.txt10-{docarray_version}.bin')
        elif OUTPUT_MODALITY == 'text':
            # for now deactivated sample images for text
            output_modality_dir = 'text'
            data_dir = root_data_dir + output_modality_dir + '/'
            da_txt = load_data(data_dir + DATA + f'.txt10-{docarray_version}.bin')

    if OUTPUT_MODALITY == 'text':
        # censor words in text incl. in custom data
        from better_profanity import profanity

        profanity.load_censor_words()

    class UI:
        about_block = """
        ### About
        This is a meme search engine using [Jina's neural search framework](https://github.com/jina-ai/jina/).
        - [Live demo](https://examples.jina.ai/memes)
        - [Play with it in a notebook](https://colab.research.google.com/github/jina-ai/workshops/blob/main/memes/meme_search.ipynb) (t-only)
        - [Repo](https://github.com/alexcg1/jina-meme-search)
        - [Dataset](https://www.kaggle.com/abhishtagatya/imgflipscraped-memes-caption-dataset)
        """

        css = """
        <style>
            .reportview-container .main .block-container{{
                max-width: 1200px;
                padding-top: 2rem;
                padding-right: 2rem;
                padding-left: 2rem;
                padding-bottom: 2rem;
            }}
            .reportview-container .main {{
                color: "#111";
                background-color: "#eee";
            }}
        </style>
        """

    def search_by_t(search_text, limit=TOP_K) -> DocumentArray:
        st.session_state.search_count += 1
        print(f'Searching by text: {search_text}')
        data = {'host': HOST, 'text': search_text, 'limit': limit}
        if PORT:
            data['port'] = PORT
        response = requests.post(URL_HOST, json=data)
        return DocumentArray.from_json(response.content)

    def search_by_file(document, limit=TOP_K) -> DocumentArray:
        """
        Wrap file in Jina Document for searching, and do all necessary conversion to make similar to indexed Docs
        """
        st.session_state.search_count += 1
        print(f"Searching by image")
        query_doc = document
        if query_doc.blob == b'':
            if query_doc.tensor is not None:
                query_doc.convert_image_tensor_to_blob()
            elif (query_doc.uri is not None) and query_doc.uri != '':
                query_doc.load_uri_to_blob()

        data = {
            'host': HOST,
            'image': base64.b64encode(query_doc.blob).decode('utf-8'),
            'limit': limit,
        }
        if PORT:
            data['port'] = PORT
        response = requests.post(URL_HOST, json=data)
        return DocumentArray.from_json(response.content)

    def convert_file_to_document(query):
        data = query.read()
        doc = Document(blob=data)
        return doc

    # Layout
    st.markdown(
        body=UI.css,
        unsafe_allow_html=True,
    )

    # design and create toggle button
    st.write(
        '<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>',
        unsafe_allow_html=True,
    )
    st.write(
        '<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-right:50px;}</style>',
        unsafe_allow_html=True,
    )
    if INPUT_MODALITY == 'image':
        media_type = st.radio(
            '',
            ["Image", 'Webcam'],
            on_change=clear_match,
        )
    elif INPUT_MODALITY == 'text':
        media_type = 'Text'

    if media_type == "Image":
        upload_c, preview_c = st.columns([12, 1])
        query = upload_c.file_uploader("")
        if query:
            doc = convert_file_to_document(query)
            st.image(doc.blob, width=160)
            st.session_state.matches = search_by_file(document=doc)
        if da_img is not None:
            st.subheader("samples:")
            img_cs = st.columns(5)
            txt_cs = st.columns(5)
            for doc, c, txt in zip(da_img, img_cs, txt_cs):
                with c:
                    st.image(doc.blob if doc.blob else doc.tensor, width=100)
                with txt:
                    if st.button('Search', key=doc.id):
                        st.session_state.matches = search_by_file(document=doc)

    elif media_type == "Text":
        query = st.text_input("", key="text_search_box")
        if query:
            st.session_state.matches = search_by_t(search_text=query)
        if st.button("Search", key="text_search"):
            st.session_state.matches = search_by_t(search_text=query)
        if da_txt is not None:
            st.subheader("samples:")
            c1, c2, c3 = st.columns(3)
            c4, c5, c6 = st.columns(3)
            for doc, col in zip(da_txt, [c1, c2, c3, c4, c5, c6]):
                with col:
                    if st.button(doc.content, key=doc.id, on_click=clear_text):
                        st.session_state.matches = search_by_t(search_text=doc.content)

    elif media_type == 'Webcam':
        snapshot = st.button('Snapshot')

        class VideoProcessor:
            snapshot: np.ndarray = None

            def recv(self, frame):
                self.snapshot = frame.to_ndarray(format="rgb24")
                return av.VideoFrame.from_ndarray(self.snapshot, format='rgb24')

        ctx = webrtc_streamer(
            key="jina-now",
            video_processor_factory=VideoProcessor,
            client_settings=WEBRTC_CLIENT_SETTINGS,
        )

        if ctx.video_processor:
            if snapshot:
                query = ctx.video_processor.snapshot
                st.image(query, width=160)
                st.session_state.snap = query
                doc = Document(tensor=query)
                doc.convert_image_tensor_to_blob()
                st.session_state.matches = search_by_file(document=doc)
            elif st.session_state.snap is not None:
                st.image(st.session_state.snap, width=160)
        else:
            clear_match()

    if st.session_state.matches:
        matches = deepcopy(st.session_state.matches)
        if st.session_state.search_count > 2:
            st.write(
                f"🔥 How did you like Jina NOW? [Please leave feedback]({SURVEY_LINK}) 🔥"
            )
        st.header('Search results')
        # Results area
        c1, c2, c3 = st.columns(3)
        c4, c5, c6 = st.columns(3)
        c7, c8, c9 = st.columns(3)
        all_cs = [c1, c2, c3, c4, c5, c6, c7, c8, c9]
        for m in matches:
            m.scores['cosine'].value = 1 - m.scores['cosine'].value
        sorted(matches, key=lambda m: m.scores['cosine'].value, reverse=True)
        matches = [
            m
            for m in matches
            if m.scores['cosine'].value > st.session_state.min_confidence
        ]
        for c, match in zip(all_cs, matches):
            match.mime_type = OUTPUT_MODALITY

            if OUTPUT_MODALITY == 'text':
                display_text = profanity.censor(match.text).replace('\n', ' ')
                body = f"<!DOCTYPE html><html><body><blockquote>{display_text}</blockquote>"
                if match.tags.get('additional_info'):
                    additional_info = match.tags.get('additional_info')
                    if type(additional_info) == str:
                        additional_info_text = additional_info
                    elif type(additional_info) == list:
                        if len(additional_info) == 1:
                            # assumes just one line containing information on text name and creator, etc.
                            additional_info_text = additional_info
                        elif len(additional_info) == 2:
                            # assumes first element is text name and second element is creator name
                            additional_info_text = (
                                f"<em>{additional_info[0]}</em> "
                                f"<small>by {additional_info[1]}</small>"
                            )

                        else:
                            additional_info_text = " ".join(additional_info)
                    body += f"<figcaption>{additional_info_text}</figcaption>"
                body += "</body></html>"
                c.markdown(
                    body=body,
                    unsafe_allow_html=True,
                )
            elif match.uri is not None:
                if match.blob != b'':
                    match.convert_blob_to_datauri()
                if match.tensor is not None:
                    match.convert_image_tensor_to_uri()
                c.image(match.convert_blob_to_datauri().uri)
        st.markdown("""---""")
        st.session_state.min_confidence = st.slider(
            'Confidence threshold',
            0.0,
            1.0,
            key='slider',
            on_change=update_conf,
        )


def update_conf():
    st.session_state.min_confidence = st.session_state.slider


def clear_match():
    st.session_state.matches = None
    st.session_state.slider = 0.0
    st.session_state.min_confidence = 0.0
    st.session_state.snap = None


def clear_text():
    st.session_state.text_search_box = ''


def load_data(data_path: str) -> DocumentArray:
    if data_path.startswith('http'):
        try:
            # TODO try except is used as workaround
            # in case load_data is called two times from two playgrounds it can happen that
            # one of the calls created the directory right after checking that it does not exist
            # this caused errors. Now the error will be ignored.
            # Can not use `exist=True` because it is not available in py3.7
            os.makedirs('data/tmp')
        except:
            pass
        url = data_path
        data_path = (
            f"data/tmp/{base64.b64encode(bytes(url, 'utf-8')).decode('utf-8')}.bin"
        )
        if not os.path.exists(data_path):
            with urlopen(url) as f:
                content = f.read()
            with open(data_path, 'wb') as f:
                f.write(content)

    try:
        da = DocumentArray.load_binary(data_path)
    except Exception:
        da = DocumentArray.load_binary(data_path, compress='gzip')
    return da


def setup_session_state():
    if 'matches' not in st.session_state:
        st.session_state.matches = None

    if 'min_confidence' not in st.session_state:
        st.session_state.min_confidence = 0.0

    if 'im' not in st.session_state:
        st.session_state.im = None

    if 'snap' not in st.session_state:
        st.session_state.snap = None

    if 'search_count' not in st.session_state:
        st.session_state.search_count = 0


if __name__ == '__main__':
    deploy_streamlit()
