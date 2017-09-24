/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.parquet.column.values.bloom;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

import org.apache.parquet.bytes.BytesUtils;
import org.apache.parquet.column.values.RandomStr;
import org.apache.parquet.io.api.Binary;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;


public class TestBloom {
  @Test
  public void testIntBloom () throws IOException {
    Bloom bloom = new Bloom(279, Bloom.HASH.MURMUR3_X64_128, Bloom.ALGORITHM.BLOCK);
    assertEquals("bloom filter size should be adjust to 512 bytes if input bytes is 279 bytes",
      bloom.getBufferedSize(), 512);

    for(int i = 0; i<10; i++) {
      bloom.insert(bloom.hash(i));
    }

    ByteArrayOutputStream baos = new ByteArrayOutputStream((int)bloom.getBufferedSize() + bloom.HEADER_SIZE);
    bloom.writeTo(baos);

    ByteBuffer bloomBuffer = ByteBuffer.wrap(baos.toByteArray());

    int length = Integer.reverseBytes(bloomBuffer.getInt());
    int hash = Integer.reverseBytes(bloomBuffer.getInt());
    int algorithm = Integer.reverseBytes(bloomBuffer.getInt());

    byte[] bitset = new byte[length];
    bloomBuffer.get(bitset);

    bloom = new Bloom(bitset, Bloom.HASH.values()[hash], Bloom.ALGORITHM.values()[algorithm]);

    for(int i = 0; i < 10; i++) {
      assertTrue(bloom.find(bloom.hash(i)));
    }
  }

  @Test
  public void testBinaryBloom() throws IOException {
    Bloom binaryBloom = new Bloom(
      0,
      Bloom.HASH.MURMUR3_X64_128,
      Bloom.ALGORITHM.BLOCK);

    List<String> strings = new ArrayList<>();
    RandomStr randomStr = new RandomStr();
    for(int i = 0; i < 100000; i++) {
      String str = randomStr.get(10);
      strings.add(str);
      binaryBloom.insert(binaryBloom.hash(Binary.fromString(str)));
    }
    binaryBloom.flush();

    ByteArrayOutputStream baos = new ByteArrayOutputStream(
      (int)binaryBloom.getBufferedSize() + binaryBloom.HEADER_SIZE);
    binaryBloom.writeTo(baos);

    ByteBuffer bloomBuffer = ByteBuffer.wrap(baos.toByteArray());

    int length = Integer.reverseBytes(bloomBuffer.getInt());
    int hash = Integer.reverseBytes(bloomBuffer.getInt());
    int algorithm = Integer.reverseBytes(bloomBuffer.getInt());

    byte[] bitset = new byte[length];
    bloomBuffer.get(bitset);

    binaryBloom = new Bloom(ByteBuffer.wrap(bitset), Bloom.HASH.values()[hash], Bloom.ALGORITHM.values()[algorithm]);

    for(int i = 0; i < strings.size(); i++) {
      assertTrue(binaryBloom.find(binaryBloom.hash(Binary.fromString(strings.get(i)))));
    }

    // exist can be true at probability 0.01.
    int exist = 0;
    for (int i = 0; i < 100000; i++) {
      String str = randomStr.get(8);
      if (binaryBloom.find(binaryBloom.hash(Binary.fromString(str)))) {
        exist ++;
      }
    }

    // exist should be probably less than 1000 according default FPP 0.01.
    assertTrue(exist < 1200);
  }

  @Test
  public void testMurmur3() throws IOException {
    byte[] testBytes1 = {0x12};
    long hash64 = Murmur3.hash64(testBytes1);
    assertEquals(hash64, -2298819731814176267l);

    byte[] testBytes2 = {0x12, 0x34, 0x56, 0x78};
    hash64 = Murmur3.hash64(testBytes2);
    assertEquals(hash64, 9026178559051508040l);

    byte[] testBytes3 = {1, 2, 3, 4, 5, 6, 7, 8};
    hash64 = Murmur3.hash64(testBytes3);
    assertEquals(hash64, -1833705417010916211l);
  }
}
