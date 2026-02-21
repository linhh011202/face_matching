"""
Notification Service
====================
Reads the FCM registration token from Firebase Realtime Database
(written by identity_service during upload/login) and sends a push
notification back to the mobile client with the processing result.

RTDB structure written by identity_service:
    /sessions/{session_id}/fcm_token  →  "<device_fcm_token>"

This service:
  1. Reads fcm_token from /sessions/{session_id}
  2. Sends an FCM push notification with the result payload
"""

import logging

from firebase_admin import db, messaging

from app.service.storage_service import _get_firebase_app

logger = logging.getLogger(__name__)


class NotificationService:
    """Send FCM push notifications based on session_id stored in RTDB."""

    def notify_signup_result(
        self,
        session_id: str,
        user_id: str,
        success: bool,
    ) -> None:
        """Send signup processing result to the mobile client via FCM."""
        self._send_notification(
            session_id=session_id,
            event="sign_up",
            user_id=user_id,
            success=success,
            title="eKYC Sign Up",
            body=(
                "Face registration completed successfully."
                if success
                else "Face registration failed. Please try again."
            ),
        )

    def notify_signin_result(
        self,
        session_id: str,
        user_id: str,
        success: bool,
    ) -> None:
        """Send signin verification result to the mobile client via FCM."""
        self._send_notification(
            session_id=session_id,
            event="sign_in",
            user_id=user_id,
            success=success,
            title="eKYC Sign In",
            body=(
                "Face verification successful."
                if success
                else "Face verification failed. Please try again."
            ),
        )

    def _send_notification(
        self,
        *,
        session_id: str,
        event: str,
        user_id: str,
        success: bool,
        title: str,
        body: str,
    ) -> None:
        """
        Core notification logic:
          1. Read FCM token from RTDB
          2. Build and send FCM message
        """
        fcm_token = self._get_fcm_token(session_id)
        if not fcm_token:
            logger.warning(
                f"No FCM token found for session_id={session_id}, skipping notification"
            )
            return

        try:
            message = messaging.Message(
                token=fcm_token,
                notification=messaging.Notification(
                    title=title,
                    body=body,
                ),
                data={
                    "event": event,
                    "session_id": session_id,
                    "user_id": user_id,
                    "success": str(success).lower(),
                },
            )
            response = messaging.send(message)
            logger.info(
                f"FCM notification sent for session_id={session_id} "
                f"(event={event}, success={success}), response={response}"
            )
        except messaging.UnregisteredError:
            logger.warning(f"FCM token is no longer valid for session_id={session_id}")
        except Exception as e:
            logger.error(
                f"Failed to send FCM notification for session_id={session_id}: {e}"
            )

    @staticmethod
    def _get_fcm_token(session_id: str) -> str | None:
        """Read FCM token from Firebase RTDB at /sessions/{session_id}/fcm_token."""
        try:
            _get_firebase_app()
            ref = db.reference(f"/sessions/{session_id}")
            data = ref.get()
            if data and isinstance(data, dict):
                return data.get("fcm_token")
            logger.warning(f"No session data in RTDB for session_id={session_id}")
            return None
        except Exception as e:
            logger.error(
                f"Failed to read FCM token from RTDB for session_id={session_id}: {e}"
            )
            return None
