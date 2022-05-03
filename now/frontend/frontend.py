import base64
import os
import sys
from copy import deepcopy
from urllib.request import urlopen

import av
import numpy as np
import streamlit as st
from docarray import DocumentArray
from jina import Client, Document
from streamlit_webrtc import ClientSettings, webrtc_streamer

WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)


root_data_dir = (
    'https://storage.googleapis.com/jina-fashion-data/data/one-line/datasets/'
)

ds_set = {
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
}


def deploy_streamlit():
    """
    We want to provide the end-to-end experience to the user.
    Please deploy a streamlit frontend on k8s/local to access the api.
    You can get the starting point for the streamlit application from alex.
    """
    setup_session_state()
    print('Run Streamlit with:', sys.argv)
    _, host, port, output_modality, data = sys.argv
    da_img = None
    da_txt = None

    # General
    TOP_K = 9
    DEBUG = os.getenv("DEBUG", False)
    DATA_DIR = "../data/images/"

    if data in ds_set:
        if output_modality == 'image':
            output_modality_dir = 'jpeg'
            data_dir = root_data_dir + output_modality_dir + '/'
            da_img, da_txt = load_data(data_dir + data + '.img10.bin'), load_data(
                data_dir + data + '.txt10.bin'
            )
        elif output_modality == 'text':
            # for now deactivated sample images for text
            output_modality_dir = 'text'
            data_dir = root_data_dir + output_modality_dir + '/'
            da_txt = load_data(data_dir + data + '.txt10.bin')

    if output_modality == 'text':
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

    def search_by_t(input, server, port, limit=TOP_K):
        print('initialize client at', server, port)
        client = Client(host=server, protocol="grpc", port=port)
        print('search text', server, port)
        response = client.search(
            Document(text=input),
            parameters={"limit": limit, 'filter': {}},
            return_results=True,
            show_progress=True,
        )

        return response[0].matches

    def search_by_file(document, server, port, limit=TOP_K):
        """
        Wrap file in Jina Document for searching, and do all necessary conversion to make similar to indexed Docs
        """
        print('connect client to ', server, port)
        client = Client(host=server, protocol="grpc", port=port)
        query_doc = document
        if query_doc.blob != b'':
            query_doc.convert_blob_to_image_tensor()
        query_doc.set_image_tensor_shape((224, 224))
        response = client.search(
            query_doc,
            parameters={"limit": limit, 'filter': {}},
            return_results=True,
            show_progress=True,
        )

        return response[0].matches

    def convert_file_to_document(query):
        data = query.read()
        doc = Document(blob=data)
        return doc

    # Layout
    st.set_page_config(page_title="NOW", page_icon='https://jina.ai/favicon.ico')

    st.markdown(
        body=UI.css,
        unsafe_allow_html=True,
    )
    col1, mid, col2 = st.columns([1, 1, 20])
    with col1:
        st.image('https://jina.ai/favicon.ico', width=60)
    with col2:
        st.header("NOW ")

    # design and create toggle button
    st.write(
        '<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>',
        unsafe_allow_html=True,
    )
    st.write(
        '<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-right:50px;}</style>',
        unsafe_allow_html=True,
    )
    if output_modality == 'image':
        media_type = st.radio(
            '',
            ["Text", "Image", 'Webcam'],
            on_change=clear_match,
        )
    elif output_modality == 'text':
        media_type = st.radio(
            '',
            ["Image", "Text", 'Webcam'],
            on_change=clear_match,
        )

    if media_type == "Image":
        upload_c, preview_c = st.columns([12, 1])
        query = upload_c.file_uploader("")
        if query:
            doc = convert_file_to_document(query)
            st.image(doc.blob, width=160)
            st.session_state.matches = search_by_file(
                document=doc, server=host, port=port
            )
        if da_img is not None:
            st.subheader("samples:")
            img_cs = st.columns(5)
            txt_cs = st.columns(5)
            for doc, c, txt in zip(da_img, img_cs, txt_cs):
                with c:
                    st.image(doc.blob if doc.blob else doc.tensor, width=100)
                with txt:
                    if st.button('Search', key=doc.id):
                        st.session_state.matches = search_by_file(
                            document=doc,
                            server=host,
                            port=port,
                        )

    elif media_type == "Text":
        query = st.text_input("", key="text_search_box")
        if query:
            st.session_state.matches = search_by_t(input=query, server=host, port=port)
        if st.button("Search", key="text_search"):
            st.session_state.matches = search_by_t(input=query, server=host, port=port)
        if da_txt is not None:
            st.subheader("samples:")
            c1, c2, c3 = st.columns(3)
            c4, c5, c6 = st.columns(3)
            for doc, col in zip(da_txt, [c1, c2, c3, c4, c5, c6]):
                with col:
                    if st.button(doc.content, key=doc.id, on_click=clear_text):
                        st.session_state.matches = search_by_t(
                            input=doc.content, server=host, port=port
                        )

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
                st.session_state.matches = search_by_file(
                    document=doc, server=host, port=port
                )
            elif st.session_state.snap is not None:
                st.image(st.session_state.snap, width=160)
        else:
            clear_match()

    if st.session_state.matches:
        matches = deepcopy(st.session_state.matches)
        st.header('Search results')
        # Results area
        c1, c2, c3 = st.columns(3)
        c4, c5, c6 = st.columns(3)
        c7, c8, c9 = st.columns(3)
        all_cs = [c1, c2, c3, c4, c5, c6, c7, c8, c9]
        # # TODO dirty hack to filter out text. Instead output modality should be passed as parameter
        # matches = [m for m in matches if m.tensor is None]
        for m in matches:
            m.scores['cosine'].value = 1 - m.scores['cosine'].value
        sorted(matches, key=lambda m: m.scores['cosine'].value, reverse=True)
        matches = [
            m
            for m in matches
            if m.scores['cosine'].value > st.session_state.min_confidence
        ]
        for c, match in zip(all_cs, matches):
            match.mime_type = output_modality

            if output_modality == 'text':
                display_text = profanity.censor(match.text)
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
            # in case load_data is called two times from two frontends it can happen that
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


if __name__ == '__main__':
    deploy_streamlit()
