#version 150

in vec2 texCoord;
in vec2 oneTexel;

uniform sampler2D DiffuseSampler;
uniform sampler2D DepthSampler;
uniform sampler2D MainSampler;
uniform sampler2D clouds;
uniform vec2 ScreenSize;

out vec4 fragColor;
#define NUMCONTROLS 26
#define THRESH 0.5
#define FUDGE 32.0
int inControl(vec2 screenCoord, float screenWidth) {
    if (screenCoord.y < 1.0) {
        float index = floor(screenWidth / 2.0) + THRESH / 2.0;
        index = (screenCoord.x - index) / 2.0;
        if (fract(index) < THRESH && index < NUMCONTROLS && index >= 0) {
            return int(index);
        }
    }
    return -1;
}

vec4 getNotControl(sampler2D inSampler, vec2 coords, bool inctrl) {
    if (inctrl) {
        return (texture(inSampler, coords - vec2(oneTexel.x, 0.0)) + texture(inSampler, coords + vec2(oneTexel.x, 0.0)) + texture(inSampler, coords + vec2(0.0, oneTexel.y))) / 3.0;
    } else {
        return texture(inSampler, coords);
    }
}

void main() {
    int inc = inControl(gl_FragCoord.xy, ScreenSize.x);

    fragColor = texture(DiffuseSampler, texCoord);
    if(inc > 0.5) {
        fragColor = texture(MainSampler, texCoord);

        }

    gl_FragDepth = texture(DepthSampler, texCoord).r;
}