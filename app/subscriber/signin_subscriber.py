import json
import logging
import signal
import sys
import uuid
from concurrent.futures import TimeoutError

from google.cloud import pubsub_v1

from app.core.config import configs
from app.core.constants import Event
from app.service.face_verification_service import FaceVerificationService

logger = logging.getLogger(__name__)


class SignInSubscriber:
    """
    Pull messages from the PubSub subscription for sign-in events.

    Expected message format:
        {
            "event": "sign_in",
            "user_id": "550e8400-e29b-41d4-a716-446655440000",
            "session_id": "abc-123",
            "timestamp": "2026-02-20T12:00:00+00:00"
        }

    On receiving a message:
      1. Parse the user_id and session_id from the message
      2. Delegate to FaceVerificationService to verify the login face
      3. ACK or NACK the message based on success/failure
    """

    def __init__(self, face_verification_service: FaceVerificationService) -> None:
        self._face_verification_svc = face_verification_service
        self._project_id = configs.GCP_PROJECT_ID
        self._subscription_id = configs.PUBSUB_SIGNIN_SUBSCRIPTION
        self._subscriber = pubsub_v1.SubscriberClient()
        self._subscription_path = self._subscriber.subscription_path(
            self._project_id, self._subscription_id
        )
        self._streaming_pull_future = None
        logger.info(
            f"SignInSubscriber initialized — subscription: {self._subscription_path}"
        )

    def _handle_message(self, message: pubsub_v1.subscriber.message.Message) -> None:
        """Process a single PubSub message."""
        logger.info(f"Received message: {message.message_id}")

        try:
            data = json.loads(message.data.decode("utf-8"))
            event = data.get("event")
            user_id_str = data.get("user_id")
            session_id = data.get("session_id")

            if event != Event.SIGN_IN:
                logger.warning(f"Ignoring unknown event: {event}")
                message.ack()
                return

            if not user_id_str:
                logger.error("Message missing 'user_id' field, acknowledging anyway")
                message.ack()
                return

            try:
                user_id = uuid.UUID(user_id_str)
            except ValueError:
                logger.error(f"Invalid user_id format: {user_id_str}")
                message.ack()
                return

            logger.info(
                f"Processing sign_in event for user_id: {user_id} (session_id={session_id})"
            )
            is_match = self._face_verification_svc.verify_user(user_id, session_id)

            if is_match:
                logger.info(f"Face verification MATCH for user_id: {user_id}")
            else:
                logger.warning(f"Face verification NOT MATCH for user_id: {user_id}")

            # Always ACK — verification result is logged/printed
            message.ack()

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in message {message.message_id}: {e}")
            message.ack()  # Don't retry malformed messages
        except Exception as e:
            logger.error(
                f"Unexpected error processing message {message.message_id}: {e}",
                exc_info=True,
            )
            message.nack()  # Retry on unexpected errors

    def start(self) -> None:
        """Start the streaming pull subscriber (blocking)."""
        logger.info(f"Starting SignInSubscriber on {self._subscription_path}...")

        flow_control = pubsub_v1.types.FlowControl(
            max_messages=configs.PUBSUB_MAX_MESSAGES,
        )

        self._streaming_pull_future = self._subscriber.subscribe(
            self._subscription_path,
            callback=self._handle_message,
            flow_control=flow_control,
        )

        # Handle graceful shutdown
        def _shutdown(signum, frame):
            logger.info("Shutdown signal received, cancelling subscriber...")
            if self._streaming_pull_future:
                self._streaming_pull_future.cancel()
            sys.exit(0)

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        logger.info("SignInSubscriber is running. Waiting for messages...")
        print("=" * 60)
        print("  Face Matching Worker — Sign In")
        print(f"  Listening on: {self._subscription_path}")
        print("  Press Ctrl+C to stop")
        print("=" * 60)

        try:
            self._streaming_pull_future.result()
        except TimeoutError:
            logger.warning("Subscriber timed out, restarting...")
            self._streaming_pull_future.cancel()
            self._streaming_pull_future.result()
        except KeyboardInterrupt:
            logger.info("Subscriber interrupted by user")
            self._streaming_pull_future.cancel()
        except Exception as e:
            logger.error(f"Subscriber error: {e}", exc_info=True)
            self._streaming_pull_future.cancel()

    def stop(self) -> None:
        """Stop the subscriber."""
        if self._streaming_pull_future:
            self._streaming_pull_future.cancel()
            logger.info("SignInSubscriber stopped")
