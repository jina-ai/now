import base64
import io
import os
from copy import deepcopy
from urllib.parse import quote, unquote
from urllib.request import urlopen

import av
import extra_streamlit_components as stx
import numpy as np
import requests
import streamlit as st
import streamlit.components.v1 as components
from better_profanity import profanity
from docarray import Document, DocumentArray
from docarray import __version__ as docarray_version
from src.constants import (
    BUTTONS,
    COOKIE_NAME,
    SURVEY_LINK,
    WEBRTC_CLIENT_SETTINGS,
    ds_set,
    root_data_dir,
)
from src.search import (
    get_query_params,
    search_by_audio,
    search_by_image,
    search_by_text,
)
from streamlit.server.server import Server
from streamlit_webrtc import webrtc_streamer
from tornado.httputil import parse_cookie

# HEADER
st.set_page_config(page_title="NOW", page_icon='https://jina.ai/favicon.ico')


def convert_file_to_document(query):
    data = query.read()
    doc = Document(blob=data)
    return doc


def load_music_examples(DATA) -> DocumentArray:
    ds_url = root_data_dir + 'music/' + DATA + f'-song5-{docarray_version}.bin'
    return load_data(ds_url)[0, 1, 4]


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def get_cookie_manager():
    return stx.CookieManager()


cookie_manager = get_cookie_manager()


def setter_cookie(val):
    cookie_manager.set(cookie=COOKIE_NAME, val=val)


def _get_all_cookies() -> dict:
    session_infos = Server.get_current()._session_info_by_id.values()
    headers = [si.ws.request.headers for si in session_infos]
    cookie_strings = [
        header_str
        for header in headers
        for k, header_str in header.get_all()
        if k == 'Cookie'
    ]
    parsed_cookies = {k: v for c in cookie_strings for k, v in parse_cookie(c).items()}

    return parsed_cookies


def get_cookie_value():
    all_cookies = _get_all_cookies()
    for k, v in all_cookies.items():
        if k == COOKIE_NAME:
            return v


def deploy_streamlit():
    """
    We want to provide the end-to-end experience to the user.
    Please deploy a streamlit playground on k8s/local to access the api.
    You can get the starting point for the streamlit application from alex.
    """
    _, mid, _ = st.columns([0.8, 1, 1])
    with open('./logo.svg', 'r') as f:
        svg = f.read()
    with mid:
        b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
        html = r'<img width="250" src="data:image/svg+xml;base64,%s"/>' % b64
        st.write(html, unsafe_allow_html=True)
    setup_session_state()

    params = get_query_params()

    login_val = get_cookie_value()
    login_details = unquote(login_val) if login_val else None
    if not login_details:
        login = True
        code = params.code
        state = params.state
        if code and state:
            resp_jwt = requests.get(
                url=f'https://api.hubble.jina.ai/v2/rpc/user.identity.grant.auto'
                f'?code={code}&state={state}'
            ).json()
            if resp_jwt and resp_jwt['code'] == 200:
                setter_cookie(resp_jwt['data'])
                login = False
            else:
                login = True

        if login:
            redirect_uri = (
                f'https://nowrun.jina.ai/?host={params.host}&input_modality={params.output_modality}'
                f'&output_modality={params.input_modality}&data={params.data}'
            )
            redirect_uri = quote(redirect_uri)
            rsp = requests.get(
                url=f'https://api.hubble.jina.ai/v2/rpc/user.identity.authorize'
                f'?provider=jina-login&response_mode=query&redirect_uri={redirect_uri}&scope=email%20profile%20openid'
            ).json()
            redirect_to = rsp['data']['redirectTo']
            st.write('')
            st.write('You are not Logged in. Please Login.')
            st.markdown(
                get_login_button(redirect_to),
                unsafe_allow_html=True,
            )
    else:
        da_img = None
        da_txt = None
        media_type = 'Text'

        da_img, da_txt = load_example_queries(
            params.data, params.output_modality, da_img, da_txt
        )

        if params.output_modality == 'text':
            # censor words in text incl. in custom data
            from better_profanity import profanity

            profanity.load_censor_words()

        setup_design()

        if params.input_modality == 'image':
            media_type = st.radio(
                '',
                ["Image", 'Webcam'],
                on_change=clear_match,
            )
        elif params.input_modality == 'text':
            media_type = 'Text'
        elif params.input_modality == 'music':
            media_type = 'Music'

        if media_type == "Image":
            render_image(da_img)

        elif media_type == "Text":
            render_text(da_txt)

        elif media_type == 'Webcam':
            render_webcam()

        elif media_type == 'Music':
            render_music_app(params.data)

        render_matches(params.output_modality)

        add_social_share_buttons()


def load_example_queries(DATA, OUTPUT_MODALITY, da_img, da_txt):
    if DATA in ds_set:
        if OUTPUT_MODALITY == 'image' or OUTPUT_MODALITY == 'video':
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
    return da_img, da_txt


def setup_design():
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


def render_image(da_img):
    upload_c, preview_c = st.columns([12, 1])
    query = upload_c.file_uploader("")
    if query:
        doc = convert_file_to_document(query)
        st.image(doc.blob, width=160)
        st.session_state.matches = search_by_image(document=doc)
    if da_img is not None:
        st.subheader("samples:")
        img_cs = st.columns(5)
        txt_cs = st.columns(5)
        for doc, c, txt in zip(da_img, img_cs, txt_cs):
            with c:
                st.image(doc.blob if doc.blob else doc.tensor, width=100)
            with txt:
                if st.button('Search', key=doc.id):
                    st.session_state.matches = search_by_image(document=doc)


def render_text(da_txt):
    query = st.text_input("", key="text_search_box")
    if query:
        st.session_state.matches = search_by_text(search_text=query)
    if st.button("Search", key="text_search"):
        st.session_state.matches = search_by_text(search_text=query)
    if da_txt is not None:
        st.subheader("samples:")
        c1, c2, c3 = st.columns(3)
        c4, c5, c6 = st.columns(3)
        for doc, col in zip(da_txt, [c1, c2, c3, c4, c5, c6]):
            with col:
                if st.button(doc.content, key=doc.id, on_click=clear_text):
                    st.session_state.matches = search_by_text(search_text=doc.content)


def render_matches(OUTPUT_MODALITY):
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

            elif OUTPUT_MODALITY == 'music':
                display_song(c, match)

            elif OUTPUT_MODALITY in ('image', 'video'):
                if match.blob != b'':
                    match.convert_blob_to_datauri()
                elif match.tensor is not None:
                    match.convert_image_tensor_to_uri()
                elif match.uri != '':
                    match.convert_uri_to_datauri()

                if match.uri != '':
                    c.image(match.uri)
            else:
                raise ValueError(f'{OUTPUT_MODALITY} not handled')

        st.markdown("""---""")
        st.session_state.min_confidence = st.slider(
            'Confidence threshold',
            0.0,
            1.0,
            key='slider',
            on_change=update_conf,
        )


def render_music_app(DATA):
    st.header('Welcome to JinaNOW music search 👋🏽')
    st.text('Upload a song to search with or select one of the examples.')
    st.text('Pro tip: You can download search results and use them to search again :)')
    query = st.file_uploader("", type=['mp3', 'wav'])
    if query:
        doc = convert_file_to_document(query)
        st.subheader('Play your song')
        st.audio(doc.blob)
        st.session_state.matches = search_by_audio(document=doc)

    else:
        columns = st.columns(3)
        music_examples = load_music_examples(DATA)

        def on_button_click(doc_id: str):
            def callback():
                st.session_state.matches = search_by_audio(music_examples[doc_id])

            return callback

        for c, song in zip(columns, music_examples):
            display_song(c, song)
            c.button('Search', on_click=on_button_click(song.id), key=song.id)


def render_webcam():
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
            st.session_state.matches = search_by_image(document=doc)
        elif st.session_state.snap is not None:
            st.image(st.session_state.snap, width=160)
    else:
        clear_match()


def add_social_share_buttons():
    # Adding social share buttons
    _, twitter, linkedin, facebook = st.columns([0.55, 0.12, 0.12, 0.12])
    for column, name in [
        (twitter, 'twitter'),
        (linkedin, 'linkedin'),
        (facebook, 'facebook'),
    ]:
        with column:
            components.html(BUTTONS[name])


def display_song(attach_to, song_doc: Document):
    attach_to.markdown(
        body=f"<!DOCTYPE html><html><body>"
        f"<p style=\"font-size: 20px; font-weight: 700; margin-bottom: -8px\">{song_doc.tags['name']}</p>"
        f"<p style=\"margin-bottom: -5px\">{song_doc.tags['artist']}</p>"
        f"<p style=\"font-size: 10px\">{' | '.join(song_doc.tags['genre_tags'][:3])}</p>"
        f"</body></html>",
        unsafe_allow_html=True,
    )
    if 'album_cover_image_url' in song_doc.tags:
        attach_to.markdown(
            body="<!DOCTYPE html><html><body>"
            f"<img/ src={song_doc.tags['album_cover_image_url']} style=\"width: 200px; height: 200px; padding: 10px 0 10px 0\">"
            "</body></html>",
            unsafe_allow_html=True,
        )
    attach_to.audio(io.BytesIO(song_doc.blob))


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


def get_login_button(url):
    return (
        f'<a href="{url}" target="_self" class="button">'
        + 'Login'
        + """<style>
                                    .button {
                                      margin-top: -50px;
                                      position: relative;
                                      overflow: hidden;
                                      -webkit-transition: background 400ms;
                                      transition: background 400ms;
                                      color: #fff;
                                      background-color: #90ee90;
                                      padding: 0.5em 0.5rem;
                                      font-family: 'Roboto', sans-serif;
                                      font-size: 1.0rem;
                                      outline: 0;
                                      border: 0;
                                      border-radius: 0.05rem;
                                      -webkit-box-shadow: 0 0 0.5rem rgba(0, 0, 0, 0.3);
                                      box-shadow: 0 0 0.5rem rgba(0, 0, 0, 0.3);
                                      cursor: pointer;
                                    }
                                    </style>
                                """
        + '</a>'
    )


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
