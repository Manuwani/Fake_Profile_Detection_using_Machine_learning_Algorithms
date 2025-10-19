import requests
import os

def fetch_instagram_profile(username):
    url = f"https://instagram-looter2.p.rapidapi.com/profile?username={username}"
    headers = {
        'x-rapidapi-key': "421eec6dfamsh687dd2a711b67a8p17e6c8jsnc4bfe668bba2",  # Replace this with your valid key
        'x-rapidapi-host': "instagram-looter2.p.rapidapi.com"
    }

    try:
        response = requests.get(url, headers=headers)
        print("Status Code:", response.status_code)
        print("Response Text:", response.text)

        if response.status_code != 200:
            return None

        data = response.json()
        if not data.get("status"):
            return None

        return {
            "id": data.get("id", -1),
            "username": data.get("username", ""),
            "full_name": data.get("full_name", ""),
            "biography": data.get("biography", ""),
            "profile_pic_url": data.get("profile_pic_url", ""),
            "follower_count": data.get("edge_followed_by", {}).get("count", 0),
            "following_count": data.get("edge_follow", {}).get("count", 0),
            "media_count": data.get("edge_owner_to_timeline_media", {}).get("count", 0),
            "is_private": int(data.get("is_private", False)),
            "is_verified": int(data.get("is_verified", False)),
            "has_anonymous_profile_picture": int(data.get("has_anonymous_profile_picture", False)),
            "has_highlight_reels": int(data.get("highlight_reel_count", 0)),
            "has_music_on_profile": int(data.get("has_music_on_profile", False)),
            "total_igtv_videos": int(data.get("total_igtv_videos", 0)),
            "total_clips_count": int(data.get("total_clips_count", 0)),
            "total_ar_effects": int(data.get("has_ar_effects", False)),
            "is_joined_recently": int(data.get("is_joined_recently", False)),
        }

    except Exception as e:
        print("Exception occurred:", str(e))
        return None


def extract_features_from_profile(profile):
    if not profile:
        return {}

    return {
        "id": int(profile.get("id", -1)),
        "media_count": int(profile.get("media_count", 0)),
        "edge_followed_by": int(profile.get("follower_count", 0)),
        "edge_follow": int(profile.get("following_count", 0)),
        "default_profile": int(profile.get("is_private", 0)),
        "protected": int(profile.get("is_private", 0)),
        "verified": int(profile.get("is_verified", 0)),
        "username_length": len(profile.get("username", "")),
        "full_name_length": len(profile.get("full_name", "")),
        "biography_length": len(profile.get("biography", "")),
        "has_highlight_reels": int(profile.get("has_highlight_reels", 0)),
        "has_music_on_profile": int(profile.get("has_music_on_profile", 0)),
        "total_igtv_videos": int(profile.get("total_igtv_videos", 0)),
        "total_clips_count": int(profile.get("total_clips_count", 0)),
        "total_ar_effects": int(profile.get("total_ar_effects", 0)),
        "is_joined_recently": int(profile.get("is_joined_recently", False))
    }


def save_profile_picture(url, username):
    if not url:
        return None
    try:
        response = requests.get(url)
        if response.status_code == 200:
            os.makedirs("profile_pics", exist_ok=True)
            path = f"profile_pics/{username}.jpg"
            with open(path, "wb") as f:
                f.write(response.content)
            return path
    except Exception as e:
        print("Failed to save profile picture:", str(e))
    return None
