import base64

def convert_img_url(img_url: str) -> str | None:
    try:
        with open(img_url, "rb") as file:
            content = file.read()
            if content:
                encoded_file = base64.b64encode(content).decode("utf-8")
                img_url = f"data:image/jpeg;base64,{encoded_file}"
                return img_url

            else: return None

    except:
        return None
