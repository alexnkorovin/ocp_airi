# before using S3 you need to get access to your S3 folder by get_access_to_s3() function
try:
    import client_lib
except ImportError:
    raise RuntimeError("Скрипт не предназначен для запуска вне кластера")


def get_access_to_s3(
    namespace="n-ws-f5nq8",
    access_key_id="u-ws-f5nq8-qco",
    security_key="aV2AiFDz5sIeHrZjba8hGcVqzsga1mDYN8SlDynb",
):
    """
    All needed parameters(credentials) are available in the tab "обзор хранилищ"
    In order to get it you need to press the button with 3 vertical dots and choose "Узнать credentials"
    """
    client_lib.save_aws_credentials(namespace, access_key_id, security_key)


def load_to_s3_from_nfc(s3_directory, nfc_directory, progress_bar=True):
    """
    every S3 directory or file should look like "s3://your_s3_bucket"
    every NFC directory or file should look like "/home/jovyan/your_folder_or_file"
    """
    from_s3_jb = client_lib.S3CopyJob(nfc_directory, s3_directory)
    from_s3_jb.submit()
    if progress_bar:
        from_s3_jb.wait()


def load_from_s3_to_nfc(nfc_file, s3_directory, progress_bar=True):
    """
    every S3 directory or file should be look like "s3://your_s3_bucket"
    every NFC file or file should be look like "/home/jovyan/your_folder_or_file"
    """
    from_s3_jb = client_lib.S3CopyJob(s3_directory, nfc_file)
    from_s3_jb.submit()
    if progress_bar:
        from_s3_jb.wait()
