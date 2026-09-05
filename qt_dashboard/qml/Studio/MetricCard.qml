import QtQuick
import QtQuick.Layouts
import Studio

// KPI tile: label, big value, optional delta line (inverse colouring for losses).
Rectangle {
    property string label: ""
    property string value: "—"
    property string delta: ""
    property bool inverse: false        // lower is better
    property color valueColor: Theme.text

    Layout.fillWidth: true
    implicitHeight: col.implicitHeight + 20
    radius: Theme.radius
    color: Theme.card
    border.color: Theme.border
    border.width: 1

    ColumnLayout {
        id: col
        anchors.fill: parent
        anchors.margins: 10
        spacing: 3
        Text {
            text: label
            color: Theme.textMuted
            font.pixelSize: Theme.fsTiny
            elide: Text.ElideRight
            Layout.fillWidth: true
        }
        Text {
            text: value
            color: valueColor
            font.pixelSize: 19
            font.bold: true
            elide: Text.ElideRight
            Layout.fillWidth: true
        }
        Text {
            visible: delta !== ""
            text: delta
            font.pixelSize: Theme.fsTiny
            color: {
                if (delta === "") return Theme.textMuted
                var up = delta.indexOf("-") !== 0
                return (up !== inverse) ? Theme.success : Theme.danger
            }
            Layout.fillWidth: true
            elide: Text.ElideRight
        }
    }
}
