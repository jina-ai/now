from streamlit_webrtc import ClientSettings


class Parameters:
    host: str
    port: str = None
    input_modality: str
    output_modality: str
    data: str


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
TOP_K = 9

BUTTONS = {
    'twitter': """
                <a href="https://twitter.com/share?ref_src=twsrc%5Etfw" class="twitter-share-button"
                Tweet
                </a>
                <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
            """,
    'linkedin': """
                <a href="https://www.linkedin.com/sharing/share-offsite/?url=https://now.jina.ai"
                class="linkedin-share-button"
                rel="noreferrer noopener" when using target="_blank">
                </a>
                <script src="https://platform.linkedin.com/in.js" type="text/javascript">lang: en_US</script>
                <script type="IN/Share" data-url="https://now.jina.ai"></script>
            """,
    'facebook': """
                <a href="https://www.facebook.com/sharer.php?u=https://now.jina.ai" class="facebook-share-button"
                rel="noreferrer noopener" when using target="_blank">
                </a>
                <div id="fb-root"></div>
                <script async defer crossorigin="anonymous"
                src="https://connect.facebook.net/en_GB/sdk.js#xfbml=1&version=v14.0" nonce="kquhy3fp"></script>
                <div class="fb-share-button" data-href="https://now.jina.ai" data-layout="button" data-size="small">
                <a target="_blank"
                 href="https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Fnow.jina.ai%2F&amp;src=sdkpreparse"
                  class="fb-xfbml-parse-ignore">Share</a></div>
            """,
}
