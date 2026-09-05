import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import Studio

// Numeric entry that also handles fractional values (lr0, weight decay) by
// scaling into SpinBox's integer domain.
SpinBox {
    id: ctl
    property int decimals: 0
    property real factor: Math.pow(10, decimals)
    property real realValue: value / factor
    property real realFrom: 0
    property real realTo: 100
    property real realStep: 1

    Layout.fillWidth: true
    implicitHeight: 34
    editable: true
    from: Math.round(realFrom * factor)
    to: Math.round(realTo * factor)
    stepSize: Math.max(1, Math.round(realStep * factor))
    font.pixelSize: Theme.fsBody

    function setRealValue(v) { value = Math.round(v * factor) }

    validator: DoubleValidator {
        bottom: ctl.realFrom
        top: ctl.realTo
        decimals: ctl.decimals
        notation: DoubleValidator.StandardNotation
    }
    textFromValue: function(value, locale) {
        return Number(value / factor).toLocaleString(locale, 'f', decimals)
    }
    valueFromText: function(text, locale) {
        return Math.round(Number.fromLocaleString(locale, text) * factor)
    }

    contentItem: TextInput {
        text: ctl.displayText
        color: Theme.text
        font: ctl.font
        selectByMouse: true
        horizontalAlignment: Qt.AlignLeft
        verticalAlignment: Qt.AlignVCenter
        leftPadding: 10
        readOnly: !ctl.editable
        validator: ctl.validator
        inputMethodHints: Qt.ImhFormattedNumbersOnly
    }
    background: Rectangle {
        radius: 8
        color: Theme.panel
        border.color: ctl.activeFocus ? Theme.borderStrong : Theme.border
        border.width: 1
    }
    up.indicator: Rectangle {
        x: ctl.width - width - 4
        y: 4
        width: 22; height: ctl.height / 2 - 5
        radius: 5
        color: ctl.up.pressed ? Theme.cardHover : "transparent"
        Text { anchors.centerIn: parent; text: "+"; color: Theme.textMuted; font.pixelSize: 13 }
    }
    down.indicator: Rectangle {
        x: ctl.width - width - 4
        y: ctl.height / 2 + 1
        width: 22; height: ctl.height / 2 - 5
        radius: 5
        color: ctl.down.pressed ? Theme.cardHover : "transparent"
        Text { anchors.centerIn: parent; text: "−"; color: Theme.textMuted; font.pixelSize: 13 }
    }
}
