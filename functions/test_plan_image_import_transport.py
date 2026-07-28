import os
from unittest.mock import Mock, patch

import requests

from evo_firebase import _load_sa_dict
from main import (
    _download_plan_image_import_urls,
    _sanitize_plan_image_import_urls,
)


def _download_url(object_path: str, *, host: str = "firebasestorage.googleapis.com") -> str:
    encoded_path = object_path.replace("/", "%2F")
    return (
        f"https://{host}/v0/b/evoforluanching.appspot.com/o/{encoded_path}"
        "?alt=media&token=test-token"
    )


def test_plan_image_import_accepts_verified_users_temporary_storage_url():
    parts = _sanitize_plan_image_import_urls(
        [_download_url("user-123/ai-plan-import/source.jpg")],
        "user-123",
    )

    assert parts == [{
        "type": "image_url",
        "image_url": {
            "url": _download_url("user-123/ai-plan-import/source.jpg"),
            "detail": "high",
        },
        "_firebase_bucket": "evoforluanching.appspot.com",
        "_firebase_object_path": "user-123/ai-plan-import/source.jpg",
    }]


def test_plan_image_import_accepts_new_top_level_prefix():
    """Uploads under ai-plan-import/<uid>/... (sweepable by lifecycle) are valid."""
    parts = _sanitize_plan_image_import_urls(
        [_download_url("ai-plan-import/user-123/source.jpg")],
        "user-123",
    )

    assert len(parts) == 1
    assert parts[0]["_firebase_object_path"] == "ai-plan-import/user-123/source.jpg"


def test_plan_image_import_rejects_other_hosts_users_and_folders():
    urls = [
        _download_url("user-123/ai-plan-import/source.jpg", host="example.com"),
        _download_url("another-user/ai-plan-import/source.jpg"),
        _download_url("ai-plan-import/another-user/source.jpg"),
        _download_url("user-123/profile/source.jpg"),
    ]

    assert _sanitize_plan_image_import_urls(urls, "user-123") == []


def test_plan_image_import_downloads_bounded_supported_images():
    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.headers = {"content-type": "image/jpeg"}
    response.raise_for_status.return_value = None
    response.iter_content.return_value = [b"jpeg-bytes"]

    with patch("main.requests.get", return_value=response):
        parts, skipped = _download_plan_image_import_urls([{
            "image_url": {"url": _download_url("user-123/ai-plan-import/source.jpg")}
        }])

    assert skipped == 0
    assert parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_plan_image_import_reads_sanitized_storage_objects_with_evo_admin():
    sanitized = _sanitize_plan_image_import_urls(
        [_download_url("user-123/ai-plan-import/source.jpg")],
        "user-123",
    )
    blob = Mock(size=len(b"jpeg-bytes"), content_type="image/jpeg")
    blob.download_as_bytes.return_value = b"jpeg-bytes"
    bucket = Mock()
    bucket.blob.return_value = blob

    with (
        patch("evo_firebase.get_evo_app", return_value=object()),
        patch("main.storage.bucket", return_value=bucket),
    ):
        parts, skipped = _download_plan_image_import_urls(sanitized)

    # The server reads then deletes the temp object (server owns its lifecycle),
    # so blob() is addressed twice for the same path: once to read, once to
    # delete. The client no longer deletes these, which used to race the read.
    bucket.blob.assert_any_call("user-123/ai-plan-import/source.jpg")
    blob.reload.assert_called_once_with(timeout=10)
    blob.delete.assert_called_once_with(timeout=10)
    assert skipped == 0
    assert parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_plan_image_import_uses_download_token_url_after_admin_failure():
    sanitized = _sanitize_plan_image_import_urls(
        [_download_url("user-123/ai-plan-import/source.jpg")],
        "user-123",
    )
    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.headers = {"content-type": "image/jpeg"}
    response.raise_for_status.return_value = None
    response.iter_content.return_value = [b"jpeg-bytes"]

    with (
        patch("evo_firebase.get_evo_app", return_value=object()),
        patch("main.storage.bucket", side_effect=RuntimeError("no cross-project access")),
        patch("main.requests.get", return_value=response) as get,
    ):
        parts, skipped = _download_plan_image_import_urls(
            sanitized,
            firebase_id_token="verified-user-token",
        )

    assert skipped == 0
    assert get.call_args.kwargs["headers"] is None
    assert parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_plan_image_import_retries_with_verified_users_storage_token():
    sanitized = _sanitize_plan_image_import_urls(
        [_download_url("user-123/ai-plan-import/source.jpg")],
        "user-123",
    )
    forbidden = Mock()
    forbidden.__enter__ = Mock(return_value=forbidden)
    forbidden.__exit__ = Mock(return_value=False)
    forbidden_error = requests.HTTPError(response=Mock(status_code=403))
    forbidden.raise_for_status.side_effect = forbidden_error

    success = Mock()
    success.__enter__ = Mock(return_value=success)
    success.__exit__ = Mock(return_value=False)
    success.headers = {"content-type": "image/jpeg"}
    success.raise_for_status.return_value = None
    success.iter_content.return_value = [b"jpeg-bytes"]

    with (
        patch("evo_firebase.get_evo_app", return_value=object()),
        patch("main.storage.bucket", side_effect=RuntimeError("no cross-project access")),
        patch("main.requests.get", side_effect=[forbidden, success]) as get,
    ):
        parts, skipped = _download_plan_image_import_urls(
            sanitized,
            firebase_id_token="verified-user-token",
        )

    assert skipped == 0
    assert get.call_args_list[0].kwargs["headers"] is None
    assert get.call_args_list[1].kwargs["headers"] == {
        "Authorization": "Firebase verified-user-token",
    }
    assert parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_plan_image_import_falls_back_to_largest_resized_variant():
    """When storage-resize-images deleted the original, read the biggest variant."""
    sanitized = _sanitize_plan_image_import_urls(
        [_download_url("ai-plan-import/user-123/pic.jpg")],
        "user-123",
    )
    original = Mock()
    original.reload.side_effect = Exception("404 No such object")

    variant_small = Mock(size=1000, content_type="image/jpeg")
    variant_small.name = "ai-plan-import/user-123/pic_200x200.jpeg"
    variant_large = Mock(size=5000, content_type="image/jpeg")
    variant_large.name = "ai-plan-import/user-123/pic_1028x1028.jpeg"
    variant_large.download_as_bytes.return_value = b"resized-bytes"

    bucket = Mock()
    bucket.blob.return_value = original
    bucket.list_blobs.return_value = [variant_small, variant_large]

    with (
        patch("evo_firebase.get_evo_app", return_value=object()),
        patch("main.storage.bucket", return_value=bucket),
    ):
        parts, skipped = _download_plan_image_import_urls(sanitized)

    assert skipped == 0
    # The 1028x1028 variant (largest area) is chosen over the 200x200 one.
    variant_large.download_as_bytes.assert_called_once()
    variant_small.download_as_bytes.assert_not_called()
    assert parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_plan_image_import_skips_absent_object_instead_of_failing():
    """A 404 (single-use temp object already gone) must be skipped, not fatal."""
    sanitized = _sanitize_plan_image_import_urls(
        [_download_url("user-123/ai-plan-import/gone.jpg")],
        "user-123",
    )
    not_found = Mock()
    not_found.__enter__ = Mock(return_value=not_found)
    not_found.__exit__ = Mock(return_value=False)
    not_found.raise_for_status.side_effect = requests.HTTPError(
        response=Mock(status_code=404)
    )

    with (
        patch("evo_firebase.get_evo_app", return_value=object()),
        patch("main.storage.bucket", side_effect=RuntimeError("no cross-project access")),
        patch("main.requests.get", return_value=not_found),
    ):
        parts, skipped = _download_plan_image_import_urls(
            sanitized,
            firebase_id_token="verified-user-token",
        )

    assert parts == []
    assert skipped == 1


def test_evo_firebase_accepts_legacy_single_quoted_service_account_secret():
    legacy = """{'type': 'service_account',
'project_id': 'evoforluanching',
'private_key': '-----BEGIN PRIVATE KEY-----
QUJD
-----END PRIVATE KEY-----
'}"""
    with patch.dict(
        os.environ,
        {"EVO_FIREBASE_SERVICE_ACCOUNT_JSON": legacy},
    ):
        parsed = _load_sa_dict()

    assert parsed == {
        "type": "service_account",
        "project_id": "evoforluanching",
        "private_key": (
            "-----BEGIN PRIVATE KEY-----\n"
            "QUJD\n"
            "-----END PRIVATE KEY-----\n"
        ),
    }


if __name__ == "__main__":
    test_plan_image_import_accepts_verified_users_temporary_storage_url()
    test_plan_image_import_accepts_new_top_level_prefix()
    test_plan_image_import_rejects_other_hosts_users_and_folders()
    test_plan_image_import_downloads_bounded_supported_images()
    test_plan_image_import_reads_sanitized_storage_objects_with_evo_admin()
    test_plan_image_import_uses_download_token_url_after_admin_failure()
    test_plan_image_import_retries_with_verified_users_storage_token()
    test_plan_image_import_skips_absent_object_instead_of_failing()
    test_evo_firebase_accepts_legacy_single_quoted_service_account_secret()
    print("plan image import transport tests passed")
