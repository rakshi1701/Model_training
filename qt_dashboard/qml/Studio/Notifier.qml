pragma Singleton
import QtQuick

// Pages raise toasts through this; Main.qml is the only listener.
QtObject {
    signal notified(string message, string kind)
    function notify(message, kind) { notified(message, kind || "info") }
}
