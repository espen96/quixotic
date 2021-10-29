#version 150

in vec2 texCoord;
in vec2 oneTexel;
in mat4 gbufferProjection;
in mat4 gbufferProjectionInverse;
in mat4 gbufferModelView;
in mat4 gbufferModelViewInverse;
in mat4 gbufferPreviousProjection;
in mat4 gbufferPreviousModelView;
in mat4 projInv;
in mat4 projInv2;
in vec3 currChunkOffset;
in vec3 prevChunkOffset;
in float near;
in float far;
in vec3 prevPosition;
in mat4 wgbufferModelViewInverse;
in float overworld;
in float end;
uniform sampler2D DiffuseSampler;
uniform sampler2D CurrentFrameDepthSampler;
uniform vec2 ScreenSize;
uniform sampler2D PreviousFrameSampler;
uniform sampler2D PreviousFrameDepthSampler;

out vec4 fragColor;

#define TAA_OFFCENTER_REJECTION 0.5 // [0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1]
#define TAA_USE_CLOSEST_DEPTH
#define TAA_HISTORY_WEIGHT 0.15 // [0.95 0.99]
vec3 calculateWorldPos(float depth, vec2 texCoord, mat4 projMat, mat4 modelViewMat) {

    vec4 clip = vec4(texCoord * 2 - 1, depth, 1);
    vec4 viewSpace = inverse(projMat) * clip;
    viewSpace /= viewSpace.w;
    return (inverse(modelViewMat) * viewSpace).xyz;
}

float MinDepth3x3(vec2 uv) {
    float minDepth = texture(CurrentFrameDepthSampler, uv).x;
    for(int x = -1; x <= 1; ++x) {
        for(int y = -1; y <= 1; ++y) {
            if(x == 0 && y == 0) {
                continue;
            }
            minDepth = min(minDepth, texture(CurrentFrameDepthSampler, uv + oneTexel * vec2(x, y)).x);
        }
    }
    return minDepth;
}
vec3 ViewSpaceFromScreenSpace(vec3 pos, mat4 inverseProjection) {
    pos = pos * 2.0 - 1.0;
	#ifdef TAA
    pos.xy -= taaOffset;
	#endif

    vec3 viewPosition = vec3(vec2(inverseProjection[0].x, inverseProjection[1].y) * pos.xy + inverseProjection[3].xy, inverseProjection[3].z);
    viewPosition /= inverseProjection[2].w * pos.z + inverseProjection[3].w;

    return viewPosition;
}
vec3 ScreenSpaceFromViewSpace(vec3 viewPosition, mat4 projection) {
    vec3 screenPosition = vec3(projection[0].x, projection[1].y, projection[2].z) * viewPosition + projection[3].xyz;
	#ifdef TAA
    screenPosition.xy -= taaOffset * viewPosition.z;
	#endif
    return screenPosition * (0.5 / -viewPosition.z) + 0.5;
}

float luma(vec3 color) {
    return dot(color, vec3(0.21, 0.72, 0.07));
}
// Min or max component of a vector
// Float versions are here for less work when something might be a vector or scalar depending on settings
float MaxOf(float v) {
    return v;
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

#define diagonal3(m) vec3((m)[0].x, (m)[1].y, m[2].z)
#define  projMAD(m, v) (diagonal3(m) * (v) + (m)[3].xyz)

#define  projMAD2(m, v) (diagonal3(m) * (v) + vec3(0,0,m[3].b))
vec4 backProject(vec4 vec) {
    vec4 tmp = wgbufferModelViewInverse * vec;
    return tmp / tmp.w;
}
vec3 toClipSpace3(vec3 viewSpacePosition) {
    return projMAD2(gbufferProjection, viewSpacePosition) / -viewSpacePosition.z * 0.5 + 0.5;
}
vec3 toScreenSpace(vec3 p) {
    vec4 iProjDiag = vec4(gbufferProjectionInverse[0].x, gbufferProjectionInverse[1].y, gbufferProjectionInverse[2].zw);
    vec3 p3 = p * 2. - 1.;
    vec4 fragposition = iProjDiag * p3.xyzz + gbufferProjectionInverse[3];
    return fragposition.xyz / fragposition.w;
}

vec3 toClipSpace3Prev(vec3 viewSpacePosition) {
    return projMAD2(gbufferPreviousProjection, viewSpacePosition) / -viewSpacePosition.z * 0.5 + 0.5;
}
vec3 tonemap(vec3 col) {
    return col / (1 + luma(col));
}
vec3 invTonemap(vec3 col) {
    return col / (1 - luma(col));
}
	// Bicubic Catmull-Rom texture filter
	// Implementation directly based on Vulkan spec, albeit optimized
vec4 CatRom(float x) {
    vec4 vec = vec4(1.0, x, x * x, x * x * x);
    const mat4 matrix = mat4(0, 2, 0, 0, -1, 0, 1, 0, 2, -5, 4, -1, -1, 3, -3, 1);
    return (1.0 / 2.0) * matrix * vec;
}
vec4 TextureBicubicCatRom(sampler2D sampler, vec2 uv) {
    uv = ScreenSize * uv - 0.5;
		/*
		vec2 i = floor(uv);
		vec2 f = fract(uv);
		//*/ vec2 i, f = modf(uv, i);

    uv = oneTexel * i;

    vec4 weightsX = CatRom(f.x);
    vec4 weightsY = CatRom(f.y);
    vec2 w12 = vec2(weightsX[1] + weightsX[2], weightsY[1] + weightsY[2]);

    float cx = weightsX[2] / w12.x;
    float cy = weightsY[2] / w12.y;
    vec2 uv12 = uv + oneTexel * (0.5 + vec2(cx, cy));

    vec2 uv0 = uv - 0.5 * oneTexel;
    vec2 uv3 = uv + 2.5 * oneTexel;

    vec4 result = (weightsX[0] * weightsY[0]) * texture(sampler, uv0);
    result += (w12.x * weightsY[0]) * texture(sampler, vec2(uv12.x, uv0.y));
    result += (weightsX[3] * weightsY[0]) * texture(sampler, vec2(uv3.x, uv0.y));

    result += (weightsX[0] * w12.y) * texture(sampler, vec2(uv0.x, uv12.y));
    result += (w12.x * w12.y) * texture(sampler, uv12);
    result += (weightsX[3] * w12.y) * texture(sampler, vec2(uv3.x, uv12.y));

    result += (weightsX[0] * weightsY[3]) * texture(sampler, vec2(uv0.x, uv3.y));
    result += (w12.x * weightsY[3]) * texture(sampler, vec2(uv12.x, uv3.y));
    result += (weightsX[3] * weightsY[3]) * texture(sampler, uv3);

    return result;
}

vec3 Reproject(vec3 position) {

    if(position.z < 1.0) {
        bool isHand = position.z < 0.6;//texture(depthtex1, position.xy).x != texture(depthtex2, position.xy).x;
        if(!isHand) {
            position = ViewSpaceFromScreenSpace(position, gbufferProjectionInverse);
            position = mat3(gbufferModelViewInverse) * position + gbufferModelViewInverse[3].xyz;
            position = position + (currChunkOffset - prevChunkOffset);
            position = mat3(gbufferPreviousModelView) * position + gbufferPreviousModelView[3].xyz;
            position = ScreenSpaceFromViewSpace(position, gbufferPreviousProjection);
        }
    } else {
        position = ViewSpaceFromScreenSpace(position, gbufferProjectionInverse);
        position = mat3(gbufferModelViewInverse) * position + gbufferModelViewInverse[3].xyz;
        position = mat3(gbufferPreviousModelView) * position + gbufferPreviousModelView[3].xyz;
        position = ScreenSpaceFromViewSpace(position, gbufferPreviousProjection);
    }

    return position;
}
#define fsign(a)  (clamp((a)*1e35,0.,1.)*2.-1.)

float interleaved_gradientNoise() {
    return fract(52.9829189 * fract(0.06711056 * gl_FragCoord.x + 0.00583715 * gl_FragCoord.y));
}
float triangularize(float dither) {
    float center = dither * 2.0 - 1.0;
    dither = center * inversesqrt(abs(center));
    return clamp(dither - fsign(center), 0.0, 1.0);
}
vec3 fp10Dither(vec3 color, float dither) {
    const vec3 mantissaBits = vec3(6., 6., 5.);
    vec3 exponent = floor(log2(color));
    return color + dither * exp2(-mantissaBits) * exp2(exponent);
}
vec3 GetHistory(sampler2D historySampler, vec2 reprojectedPosition, inout float historyWeight) {
    if(clamp(reprojectedPosition.xy, 0, 1) != reprojectedPosition.xy) {
        historyWeight = 0.001;
        return vec3(0.0);
    }

    vec2 pixelCenterDist = 2.0 * fract(reprojectedPosition.xy * ScreenSize) - 1.0;
    vec2 tmp = 1.0 - pixelCenterDist * pixelCenterDist;
    historyWeight *= tmp.x * tmp.y * TAA_OFFCENTER_REJECTION + (1.0 - TAA_OFFCENTER_REJECTION);

    return texture(historySampler, reprojectedPosition.xy).rgb;
}

void main() {

    vec3 currColor = texture(DiffuseSampler, texCoord).rgb;
    fragColor = vec4(currColor, 1);
    float depth = texture(CurrentFrameDepthSampler, texCoord).r;
    {
        vec3 position = vec3(texCoord, MinDepth3x3(texCoord));
        vec3 current = TextureBicubicCatRom(DiffuseSampler, texCoord).rgb;

    // Reprojection based aliasing

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
        vec2 reprojectedPosition = prevTexCoord;

        float historyWeight = TAA_HISTORY_WEIGHT;

        vec3 history = GetHistory(PreviousFrameSampler, reprojectedPosition, historyWeight);

        vec3 mc = texture(DiffuseSampler, texCoord).rgb;
        vec3 tl = texture(DiffuseSampler, texCoord + oneTexel * vec2(-1, -1)).rgb;
        vec3 tc = texture(DiffuseSampler, texCoord + oneTexel * vec2(0, -1)).rgb;
        vec3 tr = texture(DiffuseSampler, texCoord + oneTexel * vec2(1, -1)).rgb;
        vec3 ml = texture(DiffuseSampler, texCoord + oneTexel * vec2(-1, 0)).rgb;
        vec3 mr = texture(DiffuseSampler, texCoord + oneTexel * vec2(1, 0)).rgb;
        vec3 bl = texture(DiffuseSampler, texCoord + oneTexel * vec2(-1, 1)).rgb;
        vec3 bc = texture(DiffuseSampler, texCoord + oneTexel * vec2(0, 1)).rgb;
        vec3 br = texture(DiffuseSampler, texCoord + oneTexel * vec2(1, 1)).rgb;

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

        history = ClipAABB(history, minRounded, avgRounded, maxRounded);

		//--//
        float test = distance(texture(DiffuseSampler, texCoord).rgb, texture(PreviousFrameSampler, reprojectedPosition.xy).rgb)*2.0;

        vec3 color = (invTonemap(mix(tonemap(current), tonemap(history), clamp(test, 0.1, 1.0))));
        if(overworld != 1)
            color = current;
        fragColor.rgb = color;
      

        //fragColor.rgb = vec3(distance(texture(DiffuseSampler, texCoord).rgb, texture(PreviousFrameSampler, reprojectedPosition.xy).rgb));
        //fragColor.rgb = vec3(test);

    }

}