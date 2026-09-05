import QtQuick
import QtQuick.Layouts
import Studio

// Label above a control, with an optional hint underneath.
ColumnLayout {
    property string label: ""
    property string hint: ""
    default property alias controlData: holder.data
    Layout.fillWidth: true
    spacing: 4

    Text {
        text: label
        visible: label !== ""
        color: Theme.textMuted
        font.pixelSize: Theme.fsSmall
        Layout.fillWidth: true
        elide: Text.ElideRight
    }
    ColumnLayout {
        id: holder
        Layout.fillWidth: true
        spacing: 4
    }
    Text {
        text: hint
        visible: hint !== ""
        color: Qt.rgba(0.58, 0.65, 0.72, 1)
        font.pixelSize: Theme.fsTiny
        wrapMode: Text.WordWrap
        Layout.fillWidth: true
    }
}
