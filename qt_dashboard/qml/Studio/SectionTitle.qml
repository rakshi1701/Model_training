import QtQuick
import QtQuick.Layouts
import Studio

ColumnLayout {
    property string title: ""
    property string subtitle: ""
    Layout.fillWidth: true
    spacing: 2
    Text {
        text: title
        color: Theme.text
        font.pixelSize: Theme.fsTitle
        font.bold: true
        Layout.fillWidth: true
        wrapMode: Text.WordWrap
    }
    Text {
        text: subtitle
        visible: subtitle !== ""
        color: Theme.textMuted
        font.pixelSize: Theme.fsSmall
        Layout.fillWidth: true
        wrapMode: Text.WordWrap
    }
}
