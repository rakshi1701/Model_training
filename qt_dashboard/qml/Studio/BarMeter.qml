import QtQuick
import QtQuick.Layouts
import Studio

// Labelled utilisation bar (CPU core, VRAM, RAM, quota …).
ColumnLayout {
    property string label: ""
    property string valueText: ""
    property real value: 0          // 0..1
    property color barColor: Theme.accent
    property int barHeight: 14
    property int labelWidth: 0      // >0 puts the label inline on the left

    Layout.fillWidth: true
    spacing: 3

    RowLayout {
        Layout.fillWidth: true
        spacing: 6
        visible: labelWidth === 0 && (label !== "" || valueText !== "")
        Text {
            text: label
            color: Theme.textMuted
            font.pixelSize: Theme.fsSmall
            Layout.fillWidth: true
            elide: Text.ElideRight
        }
        Text {
            text: valueText
            color: barColor
            font.pixelSize: Theme.fsSmall
            font.bold: true
        }
    }
    RowLayout {
        Layout.fillWidth: true
        spacing: 6
        Text {
            visible: labelWidth > 0
            text: label
            color: Theme.textMuted
            font.family: Theme.mono
            font.pixelSize: Theme.fsTiny
            Layout.preferredWidth: labelWidth
        }
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: barHeight
            radius: barHeight / 3
            color: Qt.rgba(1, 1, 1, 0.06)
            Rectangle {
                width: Math.max(0, Math.min(1, value)) * parent.width
                height: parent.height
                radius: parent.radius
                color: barColor
                Behavior on width { NumberAnimation { duration: 250 } }
            }
        }
        Text {
            visible: labelWidth > 0
            text: valueText
            color: Theme.textDim
            font.family: Theme.mono
            font.pixelSize: Theme.fsTiny
            horizontalAlignment: Text.AlignRight
            Layout.preferredWidth: 40
        }
    }
}
