#version 150

in vec2 texCoord;
uniform sampler2D DiffuseDepthSampler;
uniform sampler2D PreviousFrameSampler;
uniform sampler2D DiffuseSampler;
uniform vec2 ScreenSize;
in vec2 oneTexel;
out vec4 fragColor;

in mat4 gbufferProjection;
in mat4 gbufferProjectionInverse;
in mat4 gbufferModelView;
in mat4 gbufferModelViewInverse;
in mat4 gbufferPreviousProjection;
in mat4 gbufferPreviousModelView;
in vec3 prevPosition;

/*
    -- Vertex Engine X --

    Copyright 2020 UAA Software

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
    associated documentation files (the "Software"), to deal in the Software without restriction,
    including without limitation the rights to use, copy, modify, merge, publish, distribute,
    sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial
    portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
    NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
    OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
    CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#define VXAA_TEXTURE_CURRENT DiffuseSampler
#define VXAA_TEXTURE_PREV PreviousFrameSampler

#define VXAA_W 0
#define VXAA_E 1
#define VXAA_N 2
#define VXAA_S 3
#define VXAA_NW 0
#define VXAA_NE 1
#define VXAA_SW 2
#define VXAA_SE 3

float VXAALuma(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

float VXAALumaDiff(vec3 x, vec3 y) {
    float l1 = dot(x, vec3(0.2126, 0.7152, 0.0722));
    float l2 = dot(y, vec3(0.2126, 0.7152, 0.0722));
    return abs(l1 - l2);
}

vec3 VXAAClampHistory(vec3 history, vec4 currN[4]) {
    vec3 cmin = min(min(currN[0].rgb, currN[1].rgb), min(currN[2].rgb, currN[3].rgb));
    vec3 cmax = max(min(currN[0].rgb, currN[1].rgb), max(currN[2].rgb, currN[3].rgb));
    return vec3(clamp(history.r, cmin.r, cmax.r), clamp(history.g, cmin.g, cmax.g), clamp(history.b, cmin.b, cmax.b));
}

vec2 VXAADifferentialBlendWeight(vec4 n[4]) {
    float diffWE = VXAALumaDiff(n[VXAA_W].rgb, n[VXAA_E].rgb);
    float diffNS = VXAALumaDiff(n[VXAA_N].rgb, n[VXAA_S].rgb);
    return diffWE < diffNS ? vec2(0.5, 0.0) : vec2(0.0, 0.5);
}

vec4 VXAADifferentialBlend(vec4 n[4], vec2 w) {
    vec4 c = vec4(0.0);
    c += (n[VXAA_W] + n[VXAA_E]) * w.x;
    c += (n[VXAA_N] + n[VXAA_S]) * w.y;
    return c;
}

void VXAAUpsampleT4x(out vec4 vtex[4], vec4 current, vec4 history, vec4 currN[4], vec4 histN[4]) {
    vec4 n1[4], n2[4];

    n1[VXAA_W] = currN[VXAA_W];
    n1[VXAA_E] = current;
    n1[VXAA_N] = history;
    n1[VXAA_S] = histN[VXAA_S];

    n2[VXAA_W] = history;
    n2[VXAA_E] = histN[VXAA_E];
    n2[VXAA_N] = currN[VXAA_N];
    n2[VXAA_S] = current;

    vec4 weights = vec4(VXAADifferentialBlendWeight(n1), VXAADifferentialBlendWeight(n2));
    vtex[VXAA_NW] = history;
    vtex[VXAA_NE] = VXAADifferentialBlend(n2, weights.zw);
    vtex[VXAA_SW] = VXAADifferentialBlend(n1, weights.xy);
    vtex[VXAA_SE] = current;
}
float MinDepth3x3(vec2 uv) {
    float minDepth = texture(DiffuseDepthSampler, uv).x;
    for(int x = -1; x <= 1; ++x) {
        for(int y = -1; y <= 1; ++y) {
            if(x == 0 && y == 0) {
                continue;
            }
            minDepth = min(minDepth, texture(DiffuseDepthSampler, uv + oneTexel * vec2(x, y)).x);
        }
    }
    return minDepth;
}
vec3 GetHistory(sampler2D historySampler, vec2 reprojectedPosition, inout float historyWeight) {
    if(clamp(reprojectedPosition.xy, 0, 1) != reprojectedPosition.xy) {
        historyWeight = 0.001;
        return vec3(0.0);
    }

    vec2 pixelCenterDist = 2.0 * fract(reprojectedPosition.xy * ScreenSize) - 1.0;
    vec2 tmp = 1.0 - pixelCenterDist * pixelCenterDist;
    historyWeight *= tmp.x * tmp.y * 0.5 + (1.0 - 0.5);

    return texture(historySampler, reprojectedPosition.xy).rgb;
}

float MaxOf(vec2 v) {
    return max(v.x, v.y);
}

vec3 ClipAABB(vec3 col, vec3 minCol, vec3 avgCol, vec3 maxCol) {
    vec3 clampedCol = clamp(col, minCol, maxCol);

    if(clampedCol != col) {
        vec3 cvec = avgCol - col;

        vec3 dists = mix(maxCol - col, minCol - col, step(0.0, cvec));
        dists = clamp(dists / cvec, 0, 1);

        if(clampedCol.x == col.x) { // ignore x
            if(clampedCol.y == col.y) { // ignore x+y
                col += cvec * dists.z;
            } else if(clampedCol.z == col.z) { // ignore x+z
                col += cvec * dists.y;
            } else { // ignore x
                col += cvec * MaxOf(dists.yz);
            }
        } else if(clampedCol.y == col.y) { // ignore y
            if(clampedCol.z == col.z) { // ignore y+z
                col += cvec * dists.x;
            } else { // ignore y
                col += cvec * MaxOf(dists.xz);
            }
        } else { // ignore z
            col += cvec * MaxOf(dists.xy);
        }
    }

    return col;
}
void main() {
    float depth = texture(DiffuseDepthSampler, texCoord).r;

    // We'll recreate the current world position from the texture coord and depth sampler
    float currDepth = MinDepth3x3(texCoord);
    vec4 clip = vec4(texCoord, currDepth, 1) * 2.0 - 1.0;
    vec4 viewSpace = gbufferProjectionInverse * clip;
    viewSpace /= viewSpace.w;
    vec3 worldPos = (gbufferModelViewInverse * viewSpace).xyz;
    // Then we offset this by the amount the player moved between the two frames
    vec3 prevRelativeWorldPos = worldPos - prevPosition;

    // We can then convert this into the texture coord of the fragment in the previous frame
    vec4 prevClip = gbufferPreviousProjection * gbufferPreviousModelView * vec4(prevRelativeWorldPos, 1);
    vec2 prevTexCoord = prevClip.xy / prevClip.w * 0.5 + 0.5;
    vec2 texCoord2 = texCoord;
    // Sample scene and neighbourhood.

    vec4 current = clamp(vec4(texture(VXAA_TEXTURE_CURRENT, texCoord).rgb, 1.0), vec4(0.0f), vec4(1.0f));
    vec4 history = clamp(vec4(texture(VXAA_TEXTURE_PREV, texCoord2).rgb, 1.0), vec4(0.0f), vec4(1.0f));
    current.a = VXAALuma(current.rgb);
    history.a = VXAALuma(history.rgb);
    float offset = 1.1;
    //if(depth >= 1.0) offset = 1.5;
    vec4 currN[4];
    currN[VXAA_W].rgb = clamp(texture(VXAA_TEXTURE_CURRENT, texCoord + vec2(-offset, 0.0f) / ScreenSize).rgb, vec3(0.0f), vec3(1.0f));
    currN[VXAA_E].rgb = clamp(texture(VXAA_TEXTURE_CURRENT, texCoord + vec2(offset, 0.0f) / ScreenSize).rgb, vec3(0.0f), vec3(1.0f));
    currN[VXAA_N].rgb = clamp(texture(VXAA_TEXTURE_CURRENT, texCoord + vec2(0.0f, -offset) / ScreenSize).rgb, vec3(0.0f), vec3(1.0f));
    currN[VXAA_S].rgb = clamp(texture(VXAA_TEXTURE_CURRENT, texCoord + vec2(0.0f, -offset) / ScreenSize).rgb, vec3(0.0f), vec3(1.0f));
    currN[VXAA_W].a = VXAALuma(currN[VXAA_W].rgb);
    currN[VXAA_E].a = VXAALuma(currN[VXAA_E].rgb);
    currN[VXAA_N].a = VXAALuma(currN[VXAA_N].rgb);
    currN[VXAA_S].a = VXAALuma(currN[VXAA_S].rgb);

    vec4 histN[4];

    histN[VXAA_W].rgb = clamp(texture(VXAA_TEXTURE_PREV, texCoord2 + vec2(-offset, 0.0f) / ScreenSize).rgb, vec3(0.0f), vec3(1.0f));
    histN[VXAA_E].rgb = clamp(texture(VXAA_TEXTURE_PREV, texCoord2 + vec2(offset, 0.0f) / ScreenSize).rgb, vec3(0.0f), vec3(1.0f));
    histN[VXAA_N].rgb = clamp(texture(VXAA_TEXTURE_PREV, texCoord2 + vec2(0.0f, -offset) / ScreenSize).rgb, vec3(0.0f), vec3(1.0f));
    histN[VXAA_S].rgb = clamp(texture(VXAA_TEXTURE_PREV, texCoord2 + vec2(0.0f, -offset) / ScreenSize).rgb, vec3(0.0f), vec3(1.0f));
    histN[VXAA_W].a = VXAALuma(histN[VXAA_W].rgb);
    histN[VXAA_E].a = VXAALuma(histN[VXAA_E].rgb);
    histN[VXAA_N].a = VXAALuma(histN[VXAA_N].rgb);
    histN[VXAA_S].a = VXAALuma(histN[VXAA_S].rgb);
    history.rgb = VXAAClampHistory(history.rgb, currN);

    // Temporal checkerboard upsample pass.
    vec4 vtex[4];
    VXAAUpsampleT4x(vtex, current, history, currN, histN);

        float hweight = 0.9;
        vec3 history2 = GetHistory(VXAA_TEXTURE_PREV, texCoord2, hweight);

        vec3 mc = texture(VXAA_TEXTURE_CURRENT, texCoord).rgb;
        vec3 tl = texture(VXAA_TEXTURE_CURRENT, texCoord + oneTexel * vec2(-1, -1)).rgb;
        vec3 tc = texture(VXAA_TEXTURE_CURRENT, texCoord + oneTexel * vec2(0, -1)).rgb;
        vec3 tr = texture(VXAA_TEXTURE_CURRENT, texCoord + oneTexel * vec2(1, -1)).rgb;
        vec3 ml = texture(VXAA_TEXTURE_CURRENT, texCoord + oneTexel * vec2(-1, 0)).rgb;
        vec3 mr = texture(VXAA_TEXTURE_CURRENT, texCoord + oneTexel * vec2(1, 0)).rgb;
        vec3 bl = texture(VXAA_TEXTURE_CURRENT, texCoord + oneTexel * vec2(-1, 1)).rgb;
        vec3 bc = texture(VXAA_TEXTURE_CURRENT, texCoord + oneTexel * vec2(0, 1)).rgb;
        vec3 br = texture(VXAA_TEXTURE_CURRENT, texCoord + oneTexel * vec2(1, 1)).rgb;

			// Min/Avg/Max of nearest 5 + nearest 9
        vec3 min5 = min(min(min(min(tc, ml), mc), mr), bc);
        vec3 min9 = min(min(min(min(tl, tr), min5), bl), br);
        vec3 avg5 = tc + ml + mc + mr + bc;
        vec3 avg9 = (tl + tr + avg5 + bl + br) / 9.0;
        avg5 *= 0.2;
        vec3 max5 = max(max(max(max(tc, ml), mc), mr), bc);
        vec3 max9 = max(max(max(max(tl, tr), min5), bl), br);

			// "Rounded" min/avg/max (avg of values for nearest 5 + nearest 9)
        vec3 minRounded = (min5 + min9) * 0.5;
        vec3 avgRounded = (avg5 + avg9) * 0.5;
        vec3 maxRounded = (max5 + max9) * 0.5;

        history2 = ClipAABB(history2, minRounded, avgRounded, maxRounded);

    // Average all samples.
    fragColor = clamp((vtex[VXAA_NW] + vtex[VXAA_NE] + vtex[VXAA_SW] + vtex[VXAA_SE]) * 0.25f, 0, 1);
    //if(depth >= 1.0) fragColor.rgb = clamp(mix(fragColor.rgb, history2.rgb, 0.7),0,1);
    fragColor = texture( VXAA_TEXTURE_CURRENT, texCoord );

}