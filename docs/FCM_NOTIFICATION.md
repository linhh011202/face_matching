# FCM Push Notification — eKYC Result

> **Audience**: Mobile team  
> **Last updated**: 2026-02-20

## Overview

After the mobile client uploads face photos for **Sign Up** or **Sign In**, the backend processes the images asynchronously. Once processing is complete, the server sends a **Firebase Cloud Messaging (FCM) push notification** back to the device with the result.

## Flow

```
Mobile App                  Identity Service               Face Matching Worker
    │                              │                                │
    │── POST /api/v1/ekyc/        │                                │
    │   upload-photos             │                                │
    │   (photos + fcm_token)      │                                │
    │                              │                                │
    │◄── 200 { session_id }        │                                │
    │                              │── PubSub (sign_up/sign_in) ──►│
    │                              │                                │
    │                              │                     (process faces)
    │                              │                                │
    │◄─────────── FCM Push Notification ───────────────────────────│
    │             (success: true/false)                             │
```

1. Mobile sends face photos along with the device's **FCM registration token**.
2. Identity Service returns a `session_id` immediately (HTTP 200).
3. Identity Service saves the FCM token to Firebase RTDB at `/sessions/{session_id}`.
4. Identity Service publishes a PubSub event to the Face Matching Worker.
5. Face Matching Worker processes the images and determines the result.
6. Face Matching Worker reads the FCM token from RTDB using the `session_id`.
7. Face Matching Worker sends an **FCM push notification** to the device.
8. The RTDB session entry is cleaned up after the notification is sent.

## FCM Notification Payload

### Structure

The FCM message contains both a **notification** (displayed by the OS) and a **data** payload (for app logic).

```json
{
  "notification": {
    "title": "eKYC Sign Up",
    "body": "Face registration completed successfully."
  },
  "data": {
    "event": "sign_up",
    "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "user_id": "550e8400-e29b-41d4-a716-446655440000",
    "success": "true"
  }
}
```

### Data Fields

| Field        | Type     | Description                                                                 |
|--------------|----------|-----------------------------------------------------------------------------|
| `event`      | `string` | Event type. Either `"sign_up"` or `"sign_in"`.                              |
| `session_id` | `string` | The session ID returned in the initial upload/login response (UUID format). |
| `user_id`    | `string` | The user's UUID.                                                            |
| `success`    | `string` | `"true"` if processing succeeded, `"false"` if it failed.                   |

> **Note**: FCM data payload values are always **strings**. Parse `success` as a boolean in your app code:
> ```kotlin
> val success = data["success"] == "true"
> ```
> ```swift
> let success = userInfo["success"] as? String == "true"
> ```

### Notification Text

| Event     | Success | Title           | Body                                              |
|-----------|---------|-----------------|----------------------------------------------------|
| `sign_up` | `true`  | eKYC Sign Up    | Face registration completed successfully.           |
| `sign_up` | `false` | eKYC Sign Up    | Face registration failed. Please try again.         |
| `sign_in` | `true`  | eKYC Sign In    | Face verification successful.                       |
| `sign_in` | `false` | eKYC Sign In    | Face verification failed. Please try again.         |

## Mobile Integration Guide

### 1. Obtain FCM Token

Before calling the eKYC upload or login endpoint, retrieve the device's FCM registration token.

**Android (Kotlin)**:
```kotlin
FirebaseMessaging.getInstance().token.addOnSuccessListener { token ->
    // Use this token in the API request
}
```

**iOS (Swift)**:
```swift
Messaging.messaging().token { token, error in
    guard let token = token else { return }
    // Use this token in the API request
}
```

### 2. Send Token with API Request

Include the `fcm_token` field in the multipart form data when calling:

- `POST /api/v1/ekyc/upload-photos` (Sign Up)
- `POST /api/v1/ekyc/login` (Sign In)

### 3. Handle Push Notification

Listen for incoming FCM messages and check the `data` payload:

**Android (Kotlin)**:
```kotlin
class MyFirebaseService : FirebaseMessagingService() {
    override fun onMessageReceived(remoteMessage: RemoteMessage) {
        val data = remoteMessage.data
        val event = data["event"]         // "sign_up" or "sign_in"
        val sessionId = data["session_id"]
        val userId = data["user_id"]
        val success = data["success"] == "true"

        if (event == "sign_up") {
            if (success) {
                // Navigate to success screen or show confirmation
            } else {
                // Show retry dialog
            }
        } else if (event == "sign_in") {
            if (success) {
                // Proceed with authenticated session
            } else {
                // Show verification failed message
            }
        }
    }
}
```

**iOS (Swift)**:
```swift
func application(
    _ application: UIApplication,
    didReceiveRemoteNotification userInfo: [AnyHashable: Any]
) {
    let event = userInfo["event"] as? String
    let sessionId = userInfo["session_id"] as? String
    let userId = userInfo["user_id"] as? String
    let success = userInfo["success"] as? String == "true"

    switch event {
    case "sign_up":
        success ? showSignUpSuccess() : showSignUpRetry()
    case "sign_in":
        success ? proceedToHome() : showVerificationFailed()
    default:
        break
    }
}
```

### 4. Match `session_id`

The `session_id` in the notification matches the one returned in the initial API response. Use it to correlate the notification with the correct in-progress request if the user has multiple sessions.

## Error Scenarios

| Scenario                          | Behavior                                                        |
|-----------------------------------|-----------------------------------------------------------------|
| Invalid/expired FCM token         | Notification silently skipped; logged on server side.           |
| RTDB session entry missing        | Notification skipped; may happen if session expired or was cleaned up. |
| Face processing fails             | Notification sent with `success: "false"`.                      |
| Network error during FCM send     | Logged on server; no retry (fire-and-forget).                   |

## API Reference (Quick)

### POST `/api/v1/ekyc/upload-photos`

**Request** (multipart/form-data):

| Field         | Type       | Required | Description                        |
|---------------|------------|----------|------------------------------------|
| `fcm_token`   | `string`   | Yes      | Device FCM registration token      |
| `left_faces`  | `file[]`   | Yes      | 3 left-facing face photos          |
| `right_faces` | `file[]`   | Yes      | 3 right-facing face photos         |
| `front_faces` | `file[]`   | Yes      | 3 front-facing face photos         |

**Response** (200 OK):
```json
{
  "success": true,
  "code": 0,
  "message": "Photos uploaded successfully",
  "data": {
    "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
  }
}
```

### POST `/api/v1/ekyc/login`

**Request** (multipart/form-data):

| Field        | Type       | Required | Description                        |
|--------------|------------|----------|------------------------------------|
| `email`      | `string`   | Yes      | User email address                 |
| `fcm_token`  | `string`   | Yes      | Device FCM registration token      |
| `faces`      | `file[]`   | Yes      | Exactly 3 face photos              |

**Response** (200 OK):
```json
{
  "success": true,
  "code": 0,
  "message": "Login event published successfully",
  "data": {
    "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
  }
}
```
