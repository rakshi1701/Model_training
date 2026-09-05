import QtQuick
import QtQuick.Layouts
import Studio

// Glass panel used for every grouped block in the dashboard.
Rectangle {
    id: root
    default property alias content: inner.data
    property int padding: Theme.pad
    property alias spacing: inner.spacing
    property color accentColor: "transparent"

    color: Theme.card
    radius: Theme.radius
    border.color: accentColor === "transparent" ? Theme.border : accentColor
    border.width: 1
    implicitHeight: inner.implicitHeight + padding * 2
    implicitWidth: inner.implicitWidth + padding * 2
    Layout.fillWidth: true

    ColumnLayout {
        id: inner
        anchors.fill: parent
        anchors.margins: root.padding
        spacing: 8
    }
}
