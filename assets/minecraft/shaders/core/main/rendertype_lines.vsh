#version 150

in vec3 Position;
in vec4 Color;
in vec3 Normal;
in vec2 UV2;

uniform mat4 ModelViewMat;
uniform mat4 ProjMat;
uniform float LineWidth;
uniform vec2 ScreenSize;

out float vertexDistance;
out vec4 vertexColor;
out float lmx;
out float lmy;
const float VIEW_SHRINK = 1.0 - (1.0 / 256.0);
const mat4 VIEW_SCALE = mat4(VIEW_SHRINK, 0.0, 0.0, 0.0, 0.0, VIEW_SHRINK, 0.0, 0.0, 0.0, 0.0, VIEW_SHRINK, 0.0, 0.0, 0.0, 0.0, 1.0);

out vec4 glpos;

void main() {
    vec4 linePosStart = ProjMat * VIEW_SCALE * ModelViewMat * vec4(Position, 1.0);
    vec4 linePosEnd = ProjMat * VIEW_SCALE * ModelViewMat * vec4(Position + Normal, 1.0);

    vec3 ndc1 = linePosStart.xyz / linePosStart.w;
    vec3 ndc2 = linePosEnd.xyz / linePosEnd.w;

    vec2 lineScreenDirection = normalize((ndc2.xy - ndc1.xy) * ScreenSize);
    vec2 lineOffset = vec2(-lineScreenDirection.y, lineScreenDirection.x) * LineWidth / ScreenSize;
    lmx = clamp((float(UV2.y) / 255), 0, 1);
    lmy = clamp((float(UV2.x) / 255), 0, 1);
    if(lineOffset.x < 0.0) {
        lineOffset *= -1.0;
    }

    if(gl_VertexID % 2 == 0) {
        gl_Position = vec4((ndc1 + vec3(lineOffset, 0.0)) * linePosStart.w, linePosStart.w);
    } else {
        gl_Position = vec4((ndc1 - vec3(lineOffset, 0.0)) * linePosStart.w, linePosStart.w);
    }
    glpos = gl_Position;

//    vertexDistance = length((ModelViewMat * vec4(Position, 1.0)).xyz);
    vertexColor = Color;
}
